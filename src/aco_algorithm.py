import numpy as np
from portfolio_evaluation import portfolio_performance
from constraint_enforce import enforce_constraints
from data_config import aco_params, max_seconds, time_to_convergence, min_weight, max_weight, MINACTIVE
import time
import math
import helper as hp

def aco_portfolio_optimization(expected_returns, covariance_matrix, num_ants:int=aco_params["num_ants"], alpha:float=aco_params["alpha"], evaporation:float=aco_params["evaporation"], stop_on_convergence:bool=False, min_weight:float=min_weight, max_weight:float=max_weight, minactive:int=MINACTIVE):
    """Ant Colony Optimization for portfolio optimization
    
    Args:
        expected_returns: Array or dict of expected returns
        covariance_matrix: Covariance matrix

            
    Returns:
        Tuple of (best_weights, best_sharpe, iterations_completed)
    """

    start_time = time.time()
    
    expected_returns_tickers, expected_returns_array = hp.prepare_return_lists(expected_returns)

    num_assets = len(expected_returns_array)
    
    # Initialize variables
    pheromone = np.ones(num_assets)
    best_weights = np.ones(num_assets) / num_assets  # Default to equal weights
    best_sharpe = float('-inf')
    
    # History tracking for convergence check
    history = []
    history_weights = []
    timeline = []

    # Main ACO loop
    iteration = 0
    last_increase_time = time.time()
    while True:
        iteration += 1
        all_weights = []
        all_sharpes = []
        
        # Generate solutions for each ant
        for ant in range(num_ants):
            # Sample weights proportional to pheromone levels
            weights = np.random.dirichlet((pheromone * alpha + 0.01))
            weights = enforce_constraints(weights, min_weight, max_weight, minactive)
            
            # Evaluate portfolio
            sharpe, _, _ = portfolio_performance(
                weights, expected_returns_array, covariance_matrix
            )
            
            # Ensure sharpe is a valid number (not NaN or inf)
            if np.isfinite(sharpe):
                all_weights.append(weights)
                all_sharpes.append(sharpe)
        
        # If we have valid solutions
        if all_sharpes:
            # Find the best solution in this iteration
            max_idx = np.argmax(all_sharpes)
            if all_sharpes[max_idx] > best_sharpe:
                best_sharpe = all_sharpes[max_idx]
                best_weights = all_weights[max_idx].copy()  # Make a copy to avoid reference issues
            
            # Update pheromone levels
            pheromone = (1 - evaporation) * pheromone
            
            # Apply pheromone deposit from best ant
            for i in range(num_assets):
                if np.isfinite(all_sharpes[max_idx]):  # Guard against NaN values
                    pheromone[i] += all_weights[max_idx][i] * max(0, all_sharpes[max_idx])
        



        current_time = time.time()
        if current_time - start_time >= max_seconds:
            # Break if we have reached the time limit
            num_iterations = i
            break

        if stop_on_convergence and current_time - last_increase_time > time_to_convergence:
            # Check if we have not improved for 5 seconds
            num_iterations = i + 1
            break

# only write data if increased sharpe ratio
        if iteration==0 or math.isclose(best_sharpe, history[-1] if history else float('-inf'), abs_tol=1e-6) is False:

            timeline.append(current_time - start_time)
            history.append(best_sharpe)
            history_weights.append(hp.create_best_weights_dict(expected_returns_tickers, best_weights))
    
    print(f"ACO reached iterations ({num_iterations})")

    # Add a datapoint on the last second (extrapolate the plot)
    if len(timeline) == 0 or not math.isclose(timeline[-1], max_seconds, abs_tol=1e-6):
        timeline.append(max_seconds)
        history.append(best_sharpe)
        history_weights.append(hp.create_best_weights_dict(expected_returns_tickers, best_weights))
        last_increase_time = current_time

    return best_weights, best_sharpe, num_iterations, history, history_weights, timeline