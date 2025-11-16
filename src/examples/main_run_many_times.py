import time
from datetime import datetime, timedelta
from aco_algorithm import aco_portfolio_optimization
from get_data import get_data_wrapper, get_sp500_tickers
from deap_GA import deap_portfolio_optimization
from brute_force import brute_force_portfolio_optimization
from hybrid_algorithm import hybrid_portfolio_optimization
import helper as hp
import numpy as np
import random




tickers = get_sp500_tickers()
print(f"Number of tickers: {len(tickers)}")
print(f"First 10 tickers: {tickers[:10]}")
FROM_DATE = datetime(2022, 1, 1)
TO_DATE = FROM_DATE + timedelta(days=3*365) # Three years later

# Fetch data and calculate returns, risks, and covariance matrix
print("Fetching data and calculating returns, risks, and covariance matrix...")
annualized_returns, risks, covariance_matrix = get_data_wrapper(tickers, FROM_DATE, TO_DATE)
print("Data fetched successfully.")


for _ in range(0, 50):
    try:

        print("Running ACO with convergence criteria...")
        optimal_weights_aco, optimal_sharpe_aco, aco_iterations, aco_history, aco_best_weights_history, aco_timeline = aco_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=False)

        print("\nRunning DEAP GA with convergence criteria...")
        optimal_weights_ga, optimal_sharpe_ga, ga_iterations, ga_history, ga_best_weights_history, ga_timeline = deap_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=False)

        print("Running Brute Force approach...")
        optimal_weights_bf, optimal_sharpe_bf, bf_iterations, bf_history, bf_best_weights_history, bf_timeline = brute_force_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=False)

        curr_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        random_name = random.randint(1, 1000000000)

        print("number of iterations for all algorithms:")
        print(f"ACO: {aco_iterations}, DEAP GA: {ga_iterations}, Brute Force: {bf_iterations}")

        print("time it took")
        print(f"ACO: {aco_timeline[-1]} seconds, DEAP GA: {ga_timeline[-1]} seconds, Brute Force: {bf_timeline[-1]} seconds")

        # save all data per method to files
        hp.save_to_file(aco_history, aco_best_weights_history, aco_timeline, 'aco_results_' + str(random_name) + '.txt')
        hp.save_to_file(ga_history, ga_best_weights_history, ga_timeline, 'deap_results_' + str(random_name) + '.txt')
        hp.save_to_file(bf_history, bf_best_weights_history, bf_timeline, 'bf_results_' + str(random_name) + '.txt')
        print("Results saved to files.")

    except Exception as e:
        print(f"An error occurred: {e}")
        continue
