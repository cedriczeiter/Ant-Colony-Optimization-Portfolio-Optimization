# Ant Colony Optimization for Portfolio Optimization

A comparative study of metaheuristic algorithms for solving the portfolio optimization problem, developed as part of the course **227-0707-00 S Optimization Methods for Engineers** at ETH Zürich (Spring 2025).

**Authors:** Cédric Zeiter & Alexandre Faroux

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Algorithms Implemented](#algorithms-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [License](#license)

## Overview

This project implements and compares three different optimization approaches for portfolio allocation:

1. **Ant Colony Optimization (ACO)** - A nature-inspired metaheuristic based on ant foraging behavior
2. **Genetic Algorithm (GA)** - An evolutionary algorithm using DEAP library
3. **Random Sampling (Brute Force)** - A baseline approach for comparison

The goal is to maximize the Sharpe ratio of a portfolio while respecting realistic constraints such as minimum and maximum asset weights and minimum number of active positions.

## Problem Statement

Given a set of assets with historical price data, the objective is to find the optimal portfolio allocation that maximizes the Sharpe ratio:

$$\text{Sharpe Ratio} = \frac{E[R_p] - R_f}{\sigma_p}$$

where:
- $E[R_p]$ is the expected portfolio return
- $R_f$ is the risk-free rate (3%)
- $\sigma_p$ is the portfolio volatility (standard deviation)

### Constraints

- **Maximum weight**: No single asset can exceed 30% of the portfolio
- **Minimum weight**: Active positions must be at least 0.5% of the portfolio
- **Minimum active assets**: At least 10 different stocks must be held
- **Sum constraint**: All weights must sum to 1 (fully invested)

## Algorithms Implemented

### Ant Colony Optimization (ACO)

Our ACO implementation uses:
- **Pheromone-based weight sampling**: Portfolio weights are sampled from a Dirichlet distribution influenced by pheromone levels
- **Pheromone evaporation**: Gradual decay of pheromone trails (70% retention rate)
- **Pheromone reinforcement**: Best solutions deposit pheromones proportional to their quality
- **Alpha parameter**: Controls the influence of pheromone (α = 20)
- **Colony size**: 200 ants per iteration

### Genetic Algorithm (GA)

Built using the DEAP (Distributed Evolutionary Algorithms in Python) library:
- **Population size**: 200 individuals (matching ACO colony size)
- **Selection**: Tournament selection (size = 3)
- **Crossover**: Custom two-point crossover with constraint enforcement (70% probability)
- **Mutation**: Random perturbation with constraint enforcement (30% probability, 10% per gene)
- **Elitism**: Top 3 individuals preserved across generations

### Brute Force (Random Sampling)

A baseline approach that:
- Generates random portfolio allocations
- Enforces constraints on each sample
- Tracks the best solution found
- Useful for benchmarking algorithm performance

## Project Structure

```
.
├── src/
│   ├── aco_algorithm.py           # ACO implementation
│   ├── deap_GA.py                 # Genetic algorithm using DEAP
│   ├── brute_force.py             # Random sampling baseline
│   ├── portfolio_evaluation.py    # Sharpe ratio calculation
│   ├── constraint_enforce.py      # Portfolio constraint handling
│   ├── get_data.py                # Yahoo Finance data fetching
│   ├── data_config.py             # Algorithm parameters and settings
│   ├── helper.py                  # Utility functions
│   └── examples/
│       ├── main.py                # Main comparison script
│       ├── main_run_many_times.py # Multiple runs for statistics
│       └── main_param_optimize.py # Parameter tuning experiments
├── data/                          # Historical stock price data (CSV files)
├── results/                       # Algorithm comparison results
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/cedriczeiter/Ant-Colony-Optimization-Portfolio-Optimization.git
cd Ant-Colony-Optimization-Portfolio-Optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `yfinance` - Stock data fetching
- `matplotlib` - Visualization
- `seaborn` - Statistical plotting
- `deap` - Genetic algorithm framework
- `lxml` - HTML parsing for S&P 500 list
- `openpyxl` - Excel file support

## Usage

### Basic Example

```python
from datetime import datetime, timedelta
from src.get_data import get_data_wrapper, get_sp500_tickers
from src.aco_algorithm import aco_portfolio_optimization
from src.deap_GA import deap_portfolio_optimization
from src.brute_force import brute_force_portfolio_optimization

# Fetch S&P 500 stock data
tickers = get_sp500_tickers()
from_date = datetime(2022, 1, 1)
to_date = from_date + timedelta(days=3*365)

# Get historical data and calculate statistics
annualized_returns, risks, covariance_matrix = get_data_wrapper(tickers, from_date, to_date)

# Run ACO optimization
weights_aco, sharpe_aco, iterations_aco, history_aco, _, timeline_aco = \
    aco_portfolio_optimization(annualized_returns, covariance_matrix)

# Run GA optimization
weights_ga, sharpe_ga, generations_ga, history_ga, _, timeline_ga = \
    deap_portfolio_optimization(annualized_returns, covariance_matrix)

# Run brute force baseline
weights_bf, sharpe_bf, iterations_bf, history_bf, _, timeline_bf = \
    brute_force_portfolio_optimization(annualized_returns, covariance_matrix)

print(f"ACO Sharpe Ratio: {sharpe_aco:.4f}")
print(f"GA Sharpe Ratio: {sharpe_ga:.4f}")
print(f"Brute Force Sharpe Ratio: {sharpe_bf:.4f}")
```

### Running the Main Comparison

```bash
cd src/examples
python main.py
```

This will:
1. Download historical data for all S&P 500 stocks
2. Calculate returns and covariance matrix
3. Run all three algorithms with a 30-second time limit
4. Save results to text files
5. Generate comparison plots

### Configuration

Algorithm parameters can be adjusted in `src/data_config.py`:

```python
# Portfolio Constraints
max_weight = 0.3          # Maximum 30% in any single asset
min_weight = 0.005        # Minimum 0.5% if asset is included
MINACTIVE = 10            # Minimum number of active assets

# Time limits
max_seconds = 30          # Maximum runtime per algorithm
time_to_convergence = 5   # Seconds without improvement before stopping

# ACO Parameters
aco_params = {
    "num_ants": 200,      # Number of ants per iteration
    "alpha": 20,          # Pheromone influence
    "evaporation": 0.7    # Pheromone retention rate
}

# GA Parameters
ga_params = {
    "population_size": 200,
    "mutation_mutpb": 0.3,
    "mutation_indpb": 0.1,
    "cxpb": 0.7,
    "tournsize": 3,
    "n_elite": 3
}
```

## Results

Our experiments comparing ACO, GA, and random sampling on S&P 500 portfolio optimization show:

### Performance Comparison

- **ACO** typically converges faster and finds high-quality solutions within 5-10 seconds
- **Genetic Algorithm** achieves comparable or slightly better final Sharpe ratios but may require more iterations
- **Random Sampling** serves as a baseline and demonstrates the value of structured search strategies

### Key Findings

1. **Convergence Speed**: ACO shows rapid initial improvement due to its pheromone-guided exploration
2. **Solution Quality**: Both ACO and GA significantly outperform random sampling
3. **Constraint Handling**: Custom constraint enforcement ensures all solutions remain feasible
4. **Scalability**: Algorithms successfully handle portfolios with 400+ assets (full S&P 500)

### Example Results

On a typical run with S&P 500 data (2022-2024):
- **ACO Sharpe Ratio**: ~1.2-1.5
- **GA Sharpe Ratio**: ~1.2-1.5
- **Random Sampling**: ~0.8-1.1

*Note: Results vary based on time period and market conditions*

## Technical Details

### Portfolio Evaluation

The Sharpe ratio is calculated using:
```python
def portfolio_performance(weights, expected_returns, cov_matrix):
    port_return = np.dot(weights, expected_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol
    return sharpe, port_return, port_vol
```

### Constraint Enforcement

Our constraint enforcement algorithm:
1. Clips weights to [min_weight, max_weight] range
2. Redistributes excess weight proportionally
3. Eliminates positions below minimum threshold
4. Ensures at least `MINACTIVE` positions are held
5. Normalizes weights to sum to 1

### Data Processing

Historical data is:
- Downloaded from Yahoo Finance using `yfinance`
- Cached locally in CSV files
- Processed to calculate annualized returns and covariance
- Filtered to remove invalid or incomplete data

### Convergence Criteria

Algorithms can stop either:
- After reaching the maximum time limit (30 seconds)
- After no improvement for 5 consecutive seconds (optional)

## Future Improvements

Potential extensions to this project:

- [ ] Implement hybrid ACO-GA algorithm
- [ ] Add support for transaction costs and taxes
- [ ] Implement additional risk metrics (VaR, CVaR, etc.)
- [ ] Add backtesting framework for out-of-sample validation
- [ ] Experiment with different pheromone update strategies
- [ ] Implement parallel evaluation for faster computation
- [ ] Add real-time portfolio rebalancing capabilities

## Contributing

This project was developed as part of an academic course. While we welcome feedback and suggestions, please note this is primarily an educational project.

## Acknowledgments

- **ETH Zürich** - For providing the course framework
- **DEAP Team** - For the excellent genetic algorithm library
- **Yahoo Finance** - For accessible historical stock data

## References

1. Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
2. Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91.
3. Sharpe, W. F. (1966). "Mutual Fund Performance." *The Journal of Business*, 39(1), 119-138.
4. DEAP Documentation: https://deap.readthedocs.io/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Course**: 227-0707-00 S Optimization Methods for Engineers  
**Institution**: ETH Zürich  
**Semester**: Spring 2025  
**Authors**: Cédric Zeiter & Alexandre Faroux
