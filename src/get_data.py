import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns

def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data a given interval

    :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple)
    :param start_date: Start date for historical data (datetime object)
    :param end_date: End date for historical data (datetime object)
    """

    assert isinstance(start_date, datetime), "start_date must be a datetime object"
    assert isinstance(end_date, datetime), "end_date must be a datetime object"
    assert start_date < end_date, "start_date must be earlier than end_date"
    assert end_date <= datetime.now(), "end_date must be today or earlier"

    # Save to CSV
    folder = "data"
    os.makedirs(folder, exist_ok=True)
    csv_file = os.path.join(folder, f"{ticker}_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv")

    # Check if the file already exists
    if os.path.exists(csv_file):
        print(f"Data for {ticker} already exists in {csv_file}.")
        return

    # Download historical data
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=False)
    if data.empty:
        print(f"----------> No data found for ticker {ticker}")
    

    data.to_csv(csv_file)
    print(f"Data for {ticker} saved to {csv_file}")


def calculate_annualized_return(ticker, from_date, to_date):
    """
    Reads stock data from a CSV file and calculates the annualized expected return.

    :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple)
    :return: Annualized expected return as a decimal
    """
    csv_file = f"data/{ticker}_from_{from_date.strftime('%Y-%m-%d')}_to_{to_date.strftime('%Y-%m-%d')}.csv"
    
    # Check if the file exists
    if not os.path.exists(csv_file):
        print(f"Data for {ticker} not found in {csv_file}. Please download the data first.")
        return None
    
    data = pd.read_csv(csv_file)

    # Ensure 'Adj Close' is numeric
    data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')

    # Drop rows with missing or invalid 'Adj Close' values
    data = data.dropna(subset=['Adj Close'])

    # Calculate daily returns
    data['Daily Return'] = data['Adj Close'].pct_change(fill_method=None)

    # Drop rows with missing daily returns
    data = data.dropna(subset=['Daily Return'])

    # Calculate average daily return and annualize it
    avg_daily_return = data['Daily Return'].mean()
    annualized_return = ((1 + avg_daily_return) ** 252) - 1  # 252 trading days in a year

    return annualized_return


def calculate_risk(tickers, from_date, to_date):
    """
    Calculates the annualized standard deviation (risk) for a list of stock tickers.

    :param tickers: List of stock ticker symbols
    :return: A dictionary with tickers as keys and their standard deviations as values
    """
    annualized_risks = {}
    for ticker in tickers:
        try:
            # Read data from CSV
            csv_file = f"data/{ticker}_from_{from_date.strftime('%Y-%m-%d')}_to_{to_date.strftime('%Y-%m-%d')}.csv"
            # Check if the file exists
            if not os.path.exists(csv_file):
                print(f"Data for {ticker} not found in {csv_file}. Please download the data first.")
                continue
            data = pd.read_csv(csv_file)

            # Ensure 'Adj Close' is numeric
            data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')
            data = data.dropna(subset=['Adj Close'])

            # Calculate daily returns
            data['Daily Return'] = data['Adj Close'].pct_change(fill_method=None)
            data = data.dropna(subset=['Daily Return'])

            # Calculate annualized standard deviation (risk)
            annualized_risks[ticker] = data['Daily Return'].std() * (252 ** 0.5)  #std * root(252)
        except Exception as e:
            print(f"Error processing {ticker} for annualized risk: {e}")
    return annualized_risks



def get_returns(tickers, from_date, to_date):
    """
    Fetches annualized returns for a list of stock tickers. Returns a dictionary with tickers as keys and their annualized returns as values.

    :param tickers: List of stock ticker symbols
    :return: Dictionary with ticker symbols as keys and annualized returns as values
    """
    returns = {}
    for ticker in tickers:
        try:
            returns[ticker] = calculate_annualized_return(ticker, from_date, to_date)

        except Exception as e:
            print(f"Error processing get_returns of {ticker}: {e}")
    return returns

def get_data_wrapper(tickers, from_date, to_date):
    """
    Wrapper function to download data and calculate returns and risks.

    :param tickers: List of stock ticker symbols
    :param from_date: Start date for historical data (datetime object)
    :param to_date: End date for historical data (datetime object)

    :return: A tuple containing:
        - A dictionary with tickers as keys and their annualized returns as values
        - A dictionary with tickers as keys and their annualized risks (standard deviations) as values
        - The covariance matrix as a pandas DataFrame
    """
    # First, download all data
    valid_tickers = []
    for ticker in tickers:
        try:
            download_data(ticker, from_date, to_date)
            
            # Check if the data file exists and has valid content
            csv_file = os.path.join("data", f"{ticker}_from_{from_date.strftime('%Y-%m-%d')}_to_{to_date.strftime('%Y-%m-%d')}.csv")
            if os.path.exists(csv_file):
                data = pd.read_csv(csv_file)
                if not data.empty and 'Adj Close' in data.columns and len(data) > 30:  # Ensure at least 30 days of data
                    valid_tickers.append(ticker)
                else:
                    print(f"Skipping {ticker}: insufficient or invalid data")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    print(f"Found {len(valid_tickers)} valid tickers out of {len(tickers)}")
    
    if len(valid_tickers) == 0:
        print("No valid tickers found. Check your data sources and date range.")
        return {}, {}, pd.DataFrame()
    
    # Process only valid tickers
    annualized_returns = {}
    annualized_risks = {}
    daily_returns_data = {}
    
    for ticker in valid_tickers:
        try:
            # Calculate return for this ticker
            ret = calculate_annualized_return(ticker, from_date, to_date)
            if ret is not None:
                annualized_returns[ticker] = ret
            
            # Calculate risk for this ticker
            annualized_risk = calculate_risk([ticker], from_date, to_date)
            if ticker in annualized_risk and annualized_risk[ticker] is not None:
                annualized_risks[ticker] = annualized_risk[ticker]
            
            # Get daily returns data for covariance calculation
            csv_file = f"data/{ticker}_from_{from_date.strftime('%Y-%m-%d')}_to_{to_date.strftime('%Y-%m-%d')}.csv"
            data = pd.read_csv(csv_file)
            data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')
            data = data.dropna(subset=['Adj Close'])
            data['Daily Return'] = data['Adj Close'].pct_change(fill_method=None)
            data = data.dropna(subset=['Daily Return'])
            
            if not data.empty:
                daily_returns_data[ticker] = data['Daily Return']
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    daily_returns_data = pd.DataFrame(daily_returns_data)
    
    # Only keep tickers that have both return and annualized_risk data
    common_tickers = [ticker for ticker in valid_tickers if ticker in daily_returns_data and ticker in annualized_risks]
    

    # Filter data to only include common tickers
    annualized_returns = {ticker: annualized_returns[ticker] for ticker in common_tickers if ticker in annualized_returns}
    annualized_risks = {ticker: annualized_risks[ticker] for ticker in common_tickers if ticker in annualized_risks}
    daily_returns_data = daily_returns_data[common_tickers].copy()
    
    # Calculate covariance matrix only for valid tickers
    annualized_covariance_matrix = daily_returns_data.cov() * 252  # Annualized (when assuming that evenents are indpeendent, then covariance is just addition/multiplication of single events)
    
    print(f"Final dataset contains {len(common_tickers)} stocks")
    print(f"Annualized returns data shape: {len(annualized_returns)}")
    print(f"Annulaized Risks data shape: {len(annualized_risks)}")
    print(f"Covariance matrix shape: {annualized_covariance_matrix.shape}")
    
    return annualized_returns, annualized_risks, annualized_covariance_matrix

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0] 
    tickers = sp500_table['Symbol'].tolist()
    return tickers


if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    FROM_DATE = datetime(2020, 1, 1)
    TO_DATE = FROM_DATE + 3*timedelta(days=365)  # Three years later

    print("\nDownloading data...")
    for ticker in tickers:
        download_data(ticker, FROM_DATE, TO_DATE)
    
    print("\nGet Returns and Risks...")
    # Calculate annualized returns
    returns = get_returns(tickers, FROM_DATE, TO_DATE)
    print("\nAnnualized Returns:")
    for ticker, ret in returns.items():
        if ret is not None:
            print(f"{ticker}: {ret:.2f}%")
        else:
            print(f"{ticker}: Data not available")

    # Calculate risks
    risks = calculate_risk(tickers, FROM_DATE, TO_DATE)
    print("\nRisks (Standard Deviations):")
    for ticker, risk in risks.items():
        if risk is not None:
            print(f"{ticker}: {risk:.2f}%")
        else:
            print(f"{ticker}: Data not available")

    print("\nCalculating Covariance Matrix...")
    # Calculate covariance matrix

    daily_returns = pd.DataFrame()

    for ticker in tickers:
        try:
            # Read data from CSV
            csv_file = f"data/{ticker}_from_{FROM_DATE.strftime('%Y-%m-%d')}_to_{TO_DATE.strftime('%Y-%m-%d')}.csv"
            data = pd.read_csv(csv_file)
            # Check if the file exists
            if not os.path.exists(csv_file):
                print(f"Data for {ticker} not found in {csv_file}. Please download the data first.")
                continue

            # Ensure 'Adj Close' is numeric
            data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')
            data = data.dropna(subset=['Adj Close'])

            # Calculate daily returns
            data['Daily Return'] = data['Adj Close'].pct_change(fill_method=None)
            data = data.dropna(subset=['Daily Return'])

            # Add daily returns to the DataFrame
            daily_returns[ticker] = data['Daily Return']
        except Exception as e:
            print(f"Error processing {ticker} for covariance: {e}")

    # Calculate annualized covariance matrix
    covariance_matrix = daily_returns.cov() * 252  # Annualized covariance matrix



    print("\nCovariance Matrix:")
    print(covariance_matrix)


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6)) 

    for ticker in tickers:
        csv_file = f"data/{ticker}_from_{FROM_DATE.strftime('%Y-%m-%d')}_to_{TO_DATE.strftime('%Y-%m-%d')}.csv"
        
        # Skip the first 3 rows which contain the unusual header structure
        data = pd.read_csv(csv_file, skiprows=3)
        
        # Add proper column names - there are only 7 columns, not 8
        data.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Volume']
        
        # Convert Date to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Now plot with the properly formatted DataFrame
        plt.plot(data['Date'], data['Adj Close'], label=ticker)

    plt.title('Stock Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()