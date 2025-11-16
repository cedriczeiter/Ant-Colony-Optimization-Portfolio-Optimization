import time
from datetime import datetime, timedelta
from aco_algorithm import aco_portfolio_optimization
from get_data import get_data_wrapper, get_sp500_tickers
from deap_GA import deap_portfolio_optimization
from brute_force import brute_force_portfolio_optimization
from hybrid_algorithm import hybrid_portfolio_optimization
import helper as hp
import numpy as np




tickers = get_sp500_tickers()
print(f"Number of tickers: {len(tickers)}")
print(f"First 10 tickers: {tickers[:10]}")
FROM_DATE = datetime(2022, 1, 1)
TO_DATE = FROM_DATE + timedelta(days=3*365) # Three years later

# Fetch data and calculate returns, risks, and covariance matrix
print("Fetching data and calculating returns, risks, and covariance matrix...")
annualized_returns, risks, covariance_matrix = get_data_wrapper(tickers, FROM_DATE, TO_DATE)
print("Data fetched successfully.")

# for minw in [0.005]:
#     for maxw in [0.3]:
#         if minw >= maxw:
#             print(f"Skipping invalid weight range: min_weight={minw}, max_weight={maxw}")
#             continue
#         upperbound = max(int(np.floor(1 / maxw)) + 1, 40) if maxw > 1e-10 else 40  # do not divide by zero and upperbound should be max 40
#         for mina in range(1, upperbound):
#             print(f"Running with min_weight={minw}, max_weight={maxw}, minactive={mina}")

#             try:

#                 print("Running ACO with convergence criteria...")
#                 optimal_weights_aco, optimal_sharpe_aco, aco_iterations, aco_history, aco_best_weights_history, aco_timeline = aco_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#                 print("\nRunning DEAP GA with convergence criteria...")
#                 optimal_weights_ga, optimal_sharpe_ga, ga_iterations, ga_history, ga_best_weights_history, ga_timeline = deap_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#                 # print("\nRunning Hybrid Algorithm with convergence criteria...")
#                 # optimal_weights_hybrid, optimal_sharpe_hybrid, hybrid_iterations, hybrid_history, hybrid_best_weights_history, hybrid_timeline = hybrid_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True)

#                 print("Running Brute Force approach...")
#                 optimal_weights_bf, optimal_sharpe_bf, bf_iterations, bf_history, bf_best_weights_history, bf_timeline = brute_force_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#                 curr_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#                 # save all data per method to files
#                 hp.save_to_file(aco_history, aco_best_weights_history, aco_timeline, 'aco_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#                 hp.save_to_file(ga_history, ga_best_weights_history, ga_timeline, 'deap_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#                 hp.save_to_file(bf_history, bf_best_weights_history, bf_timeline, 'bf_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#                 print("Results saved to files.")

#             except Exception as e:
#                 print(f"An error occurred: {e}")
#                 continue

# for minw in [0.0005]:
#     for maxw in np.arange(1.00, 0.00, -0.025):
#         if minw >= maxw:
#             print(f"Skipping invalid weight range: min_weight={minw}, max_weight={maxw}")
#             continue
#         for mina in [10]:
#             print(f"Running with min_weight={minw}, max_weight={maxw}, minactive={mina}")

#             try:

#                 print("Running ACO with convergence criteria...")
#                 optimal_weights_aco, optimal_sharpe_aco, aco_iterations, aco_history, aco_best_weights_history, aco_timeline = aco_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#                 print("\nRunning DEAP GA with convergence criteria...")
#                 optimal_weights_ga, optimal_sharpe_ga, ga_iterations, ga_history, ga_best_weights_history, ga_timeline = deap_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#                 # print("\nRunning Hybrid Algorithm with convergence criteria...")
#                 # optimal_weights_hybrid, optimal_sharpe_hybrid, hybrid_iterations, hybrid_history, hybrid_best_weights_history, hybrid_timeline = hybrid_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True)

#                 print("Running Brute Force approach...")
#                 optimal_weights_bf, optimal_sharpe_bf, bf_iterations, bf_history, bf_best_weights_history, bf_timeline = brute_force_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#                 curr_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#                 # save all data per method to files
#                 hp.save_to_file(aco_history, aco_best_weights_history, aco_timeline, 'aco_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#                 hp.save_to_file(ga_history, ga_best_weights_history, ga_timeline, 'deap_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#                 hp.save_to_file(bf_history, bf_best_weights_history, bf_timeline, 'bf_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#                 print("Results saved to files.")

#             except Exception as e:
#                 print(f"An error occurred: {e}")
#                 continue

for minw in np.arange(0.00, 1.00, 0.005):
    for maxw in [0.3]:
        if minw >= maxw:
            print(f"Skipping invalid weight range: min_weight={minw}, max_weight={maxw}")
            continue
        for mina in [10]:
            print(f"Running with min_weight={minw}, max_weight={maxw}, minactive={mina}")

            try:

                print("Running ACO with convergence criteria...")
                optimal_weights_aco, optimal_sharpe_aco, aco_iterations, aco_history, aco_best_weights_history, aco_timeline = aco_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

                print("\nRunning DEAP GA with convergence criteria...")
                optimal_weights_ga, optimal_sharpe_ga, ga_iterations, ga_history, ga_best_weights_history, ga_timeline = deap_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

                # print("\nRunning Hybrid Algorithm with convergence criteria...")
                # optimal_weights_hybrid, optimal_sharpe_hybrid, hybrid_iterations, hybrid_history, hybrid_best_weights_history, hybrid_timeline = hybrid_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True)

                print("Running Brute Force approach...")
                optimal_weights_bf, optimal_sharpe_bf, bf_iterations, bf_history, bf_best_weights_history, bf_timeline = brute_force_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

                curr_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


                # save all data per method to files
                hp.save_to_file(aco_history, aco_best_weights_history, aco_timeline, 'aco_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
                hp.save_to_file(ga_history, ga_best_weights_history, ga_timeline, 'deap_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
                hp.save_to_file(bf_history, bf_best_weights_history, bf_timeline, 'bf_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
                print("Results saved to files.")

            except Exception as e:
                print(f"An error occurred: {e}")
                continue

# for minw in np.arange(0.01, 1.00, 0.005):
#     for maxw in [0.3]:
#         if minw >= maxw:
#             print(f"Skipping invalid weight range: min_weight={minw}, max_weight={maxw}")
#             continue
#         for mina in [10]:
#             print(f"Running with min_weight={minw}, max_weight={maxw}, minactive={mina}")


#             # print("Running ACO with convergence criteria...")
#             # optimal_weights_aco, optimal_sharpe_aco, aco_iterations, aco_history, aco_best_weights_history, aco_timeline = aco_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#             print("\nRunning DEAP GA with convergence criteria...")
#             optimal_weights_ga, optimal_sharpe_ga, ga_iterations, ga_history, ga_best_weights_history, ga_timeline = deap_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#             # print("\nRunning Hybrid Algorithm with convergence criteria...")
#             # optimal_weights_hybrid, optimal_sharpe_hybrid, hybrid_iterations, hybrid_history, hybrid_best_weights_history, hybrid_timeline = hybrid_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True)

#             # print("Running Brute Force approach...")
#             # optimal_weights_bf, optimal_sharpe_bf, bf_iterations, bf_history, bf_best_weights_history, bf_timeline = brute_force_portfolio_optimization(annualized_returns, covariance_matrix, stop_on_convergence=True, min_weight=minw, max_weight=maxw, minactive=mina)

#             curr_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#             # save all data per method to files
#             # hp.save_to_file(aco_history, aco_best_weights_history, aco_timeline, 'aco_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#             hp.save_to_file(ga_history, ga_best_weights_history, ga_timeline, 'deap_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#             # hp.save_to_file(bf_history, bf_best_weights_history, bf_timeline, 'bf_results_' + curr_date_time + '__min_weight_' + str(minw) + "__max_weight_" + str(maxw) + "__minactive_" + str(mina) + '.txt')
#             print("Results saved to files.")


# # Plot the history of Sharpe ratios
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(aco_timeline[:aco_iterations], aco_history[:aco_iterations], marker='o', label='ACO')
# plt.plot(ga_timeline[:ga_iterations], ga_history[:ga_iterations], marker='o', label='DEAP GA')
# # plt.plot(hybrid_timeline[:hybrid_iterations], hybrid_history[:hybrid_iterations], marker='o', label='Hybrid')
# plt.plot(bf_timeline[:bf_iterations], bf_history[:bf_iterations], marker='o', label='Brute Force')
# plt.title('Different sharpe ratio histories')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Sharpe Ratio')
# plt.grid()
# plt.savefig('sharpe_ratio_history_' + curr_date_time + '.png')
# plt.show()


# hp.plot_allocations(aco_best_weights_history[-1], curr_date_time + '_aco', show_pie=True)
# hp.plot_allocations(ga_best_weights_history[-1], curr_date_time + '_ga', show_pie=True)
# # hp.plot_allocations(hybrid_best_weights_history[-1], curr_date_time + '_hybrid', show_pie=True)
# hp.plot_allocations(bf_best_weights_history[-1], curr_date_time + '_bf', show_pie=True)


