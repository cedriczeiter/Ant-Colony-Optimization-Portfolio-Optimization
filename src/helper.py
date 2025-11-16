
import matplotlib.pyplot as plt
import numpy as np


def prepare_return_lists(expected_returns):
    # Ensure expected_returns is a numpy array
    if isinstance(expected_returns, dict):
        expected_returns_array = np.array(list(expected_returns.values()))
    else:
        expected_returns_array = expected_returns

    expected_returns_tickers = np.array(list(expected_returns.keys())) if isinstance(expected_returns, dict) else None
    assert len(expected_returns_array) == len(expected_returns_tickers) if expected_returns_tickers is not None else True, "Expected returns and tickers length mismatch"

    return expected_returns_tickers, expected_returns_array

def create_best_weights_dict(expected_returns_tickers, best_weights) -> dict:
    best_weights_dict={}

    # find all weights that are not zero (numerically stable)
    for i, weight in enumerate(best_weights):
        if weight > 0.000001:
            best_weights_dict[expected_returns_tickers[i]] = weight

    return best_weights_dict


def plot_allocations(allocation_dict, identifier, show_pie=False):
    # Sort allocations by value (optional, for better visual ordering)
    allocations = dict(sorted(allocation_dict.items(), key=lambda item: item[1], reverse=True))

    tick_labels = list(allocations.keys())
    weights = list(allocations.values())

    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(tick_labels, weights, color='skyblue', edgecolor='black')
    ax.set_ylabel('Allocation Weight')
    ax.set_title('Portfolio Asset Allocation')
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_ylim(0, max(weights) * 1.2)

    # Annotate percentages on top of bars
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height, f'{weight:.2%}', 
                ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(f'portfolio_allocation_{identifier}.png')

    # Optional pie chart
    if show_pie:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.pie(weights, labels=tick_labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
        ax2.set_title('Portfolio Allocation (Pie Chart)')
        plt.tight_layout()
        plt.savefig(f'portfolio_allocation_pie_{identifier}.png')


def save_to_file(history, best_weights_history, timeline, filename):
    filename = filename if filename.endswith('.txt') else filename + '.txt'
    # put into a folder called results
    filename = 'results_correction_again/' + filename
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write("Sharpe Ratio History:\n")
        for i, sharpe in enumerate(history):
            f.write(f"{timeline[i]}: {sharpe}\n")
        
        f.write("\nBest Weights History:\n")
        for i, weights in enumerate(best_weights_history):
            f.write(f"{timeline[i]}: {weights}\n")
        
        f.write("\nTimeline:\n")
        for time in timeline:
            f.write(f"{time}\n")
    print(f"Results saved to {filename}")
    return filename 

