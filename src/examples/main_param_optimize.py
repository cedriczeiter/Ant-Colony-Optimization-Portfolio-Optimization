import numpy as np
from datetime import datetime, timedelta
from aco_algorithm import aco_portfolio_optimization
from get_data import get_data_wrapper, get_sp500_tickers
from deap_GA import deap_portfolio_optimization
import helper as hp
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run portfolio optimization with ACO or GA algorithms')
    parser.add_argument('individuals', type=int, help='Number of individuals (ants for ACO, population for GA)')
    parser.add_argument('algorithm', type=str, choices=['aco', 'ga', 'ga_mutation'], help='Algorithm to run: aco or ga or ga_muatation')
    
    args = parser.parse_args()

    tickers = get_sp500_tickers()
    print(f"Number of tickers: {len(tickers)}")
    print(f"First 10 tickers: {tickers[:10]}")
    FROM_DATE = datetime(2022, 1, 1)
    TO_DATE = FROM_DATE + timedelta(days=3*365) # Three years later

    # Fetch data and calculate returns, risks, and covariance matrix
    print("Fetching data and calculating returns, risks, and covariance matrix...")
    annualized_returns, risks, covariance_matrix = get_data_wrapper(tickers, FROM_DATE, TO_DATE)
    print("Data fetched successfully.") 

    if args.algorithm == 'aco':
        run_aco(args.individuals, annualized_returns, covariance_matrix)
    elif args.algorithm == 'ga':
        run_ga(args.individuals, annualized_returns, covariance_matrix)
    elif args.algorithm == 'ga_mutation':
        run_ga_mutation(args.individuals, annualized_returns, covariance_matrix)
    else:
        print(f"Unknown algorithm: {args.algorithm}")

    print("Optimization completed.")



def run_aco(individuals:int, annualized_returns, covariance_matrix):
    aco_results = []

    for alpha in np.arange(0, 50, 1):
        for evaporation in np.arange(0, 1, 0.01):
            print(f"Running ACO with num_ants={individuals}, alpha={alpha}, evaporation={evaporation}...")
            optimal_weights_aco, optimal_sharpe_aco, aco_iterations, aco_history, aco_best_weights_history, aco_timeline = aco_portfolio_optimization(
                annualized_returns, covariance_matrix, num_ants=individuals, alpha=alpha, evaporation=evaporation, stop_on_convergence=True
            )
            aco_results.append({
                'num_ants': individuals,
                'alpha': alpha,
                'evaporation': evaporation,
                'optimal_sharpe': optimal_sharpe_aco,
                'iterations': aco_iterations
            })

    aco_df = pd.DataFrame(aco_results)
    aco_df.to_excel(f'aco_optimization_results_pop_{individuals}.xlsx', index=False)

def run_ga(individuals:int, annualized_returns, covariance_matrix):

    ga_results = []

    for cxpb in np.arange(0, 1, 0.02):
        for indpb in np.arange(0, 1, 0.02):
            print(f"\nRunning DEAP GA with population_size={individuals}, indpb={indpb}, cxpb={cxpb}...")
            optimal_weights_ga, optimal_sharpe_ga, ga_iterations, ga_history, ga_best_weights_history, ga_timeline = deap_portfolio_optimization(
                annualized_returns, covariance_matrix, population_size=individuals, mutation_indpb=indpb, cxpb=cxpb, stop_on_convergence=True)
            ga_results.append({
                'population_size': individuals,
                'cxpb': cxpb,
                'indpb': indpb,
                'optimal_sharpe': optimal_sharpe_ga,
                'iterations': ga_iterations
            })

    ga_df = pd.DataFrame(ga_results)
    ga_df.to_excel(f'ga_optimization_results_pop_{individuals}.xlsx', index=False)


def run_ga_mutation(individuals:int, annualized_returns, covariance_matrix):

    ga_results = []

    for cxpb in np.arange(0, 1, 0.02):
        for mutpb in np.arange(0, 1, 0.02):
            print(f"\nRunning DEAP GA with population_size={individuals}, cxpb={cxpb}, mutpb={mutpb}...")
            optimal_weights_ga, optimal_sharpe_ga, ga_iterations, ga_history, ga_best_weights_history, ga_timeline = deap_portfolio_optimization(
                annualized_returns, covariance_matrix, population_size=individuals, cxpb=cxpb, mutation_mutpb=mutpb,stop_on_convergence=True)
            ga_results.append({
                'population_size': individuals,
                'cxpb': cxpb,
                'mutpb': mutpb,
                'optimal_sharpe': optimal_sharpe_ga,
                'iterations': ga_iterations
            })

    ga_df = pd.DataFrame(ga_results)
    ga_df.to_excel(f'ga_mutation_optimization_results_pop_{individuals}.xlsx', index=False)

if __name__ == "__main__":
    main()