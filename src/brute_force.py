import numpy as np
import time
from portfolio_evaluation import portfolio_performance
from constraint_enforce import enforce_constraints
from data_config import max_seconds, time_to_convergence, min_weight, max_weight, MINACTIVE
import helper as hp

def brute_force_portfolio_optimization(expected_returns, covariance_matrix, stop_on_convergence:bool=False, min_weight:float=min_weight, max_weight:float=max_weight, minactive:int=MINACTIVE):
    """Brute force approach for portfolio optimization by random sampling
    
    Args:
        expected_returns: Array or dict of expected returns
        covariance_matrix: Covariance matrix
            
    Returns:
        Tuple of (best_weights, best_sharpe, iterations_completed, history, timeline)
    """
    start_time = time.time()

    expected_returns_tickers, expected_returns_array = hp.prepare_return_lists(expected_returns)
    
    num_assets = len(expected_returns_array)
    
    # Initialize tracking variables
    best_weights = np.ones(num_assets) / num_assets  # Start with equal weights
    best_sharpe = float('-inf')
    history = []
    history_weights = []
    timeline = []

    
    # Try random portfolios
    i=0
    last_increase_time = time.time()
    while True:
        i += 1
        # Generate random weights that sum to 1
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        
        # Enforce constraints
        weights = enforce_constraints(weights, min_weight, max_weight, minactive)
        
        # Evaluate portfolio
        sharpe, _, _ = portfolio_performance(
            weights, expected_returns_array, covariance_matrix
        )
        
        # Update best solution if better
        if np.isfinite(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights.copy()
        
        current_time = time.time()

        if current_time - start_time >= max_seconds:
            # Break if we have reached the time limit
            num_iterations = i
            break

        if stop_on_convergence and current_time - last_increase_time > time_to_convergence:
            # Check if we have not improved for 5 seconds
            num_iterations = i + 1
            break
        
        
        if i==0 or not np.isclose(best_sharpe, history[-1] if history else float('-inf'), atol=1e-6):
            history.append(best_sharpe)
            timeline.append(current_time - start_time)
            history_weights.append(hp.create_best_weights_dict(expected_returns_tickers, best_weights))
            last_increase_time = current_time


    # extrapolate to end of time
    if len(timeline) == 0 or not np.isclose(timeline[-1], max_seconds, atol=1e-6):
        timeline.append(max_seconds)
        history.append(best_sharpe)
        history_weights.append(hp.create_best_weights_dict(expected_returns_tickers, best_weights))
    
    print(f"Brute force completed {num_iterations} iterations")
    return best_weights, best_sharpe, num_iterations, history, history_weights, timeline