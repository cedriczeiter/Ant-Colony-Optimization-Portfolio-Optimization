risk_free_rate = 0.03

# Portfolio Constraints
max_weight = 0.3  # Maximum 30% in any single asset
min_weight = 0.005  # Minimum 0.5% if asset is included
MINACTIVE = 10  # Minimum number of active assets in the portfolio

max_seconds = 30
time_to_convergence = 5  # seconds to wait for convergence before stopping


# ACO Parameters
aco_params = {
    "num_ants": 200,               # Matches GA population size
    "alpha": 20,                 # Influence of pheromone
    "evaporation": 0.7
}

# GA Parameters
ga_params = {
    "population_size": aco_params["num_ants"],        # Matches ACO num_ants
    "mutation_mutpb": 0.3,          # Mutation probability
    "mutation_indpb": 0.1,          # Individual gene mutation probability
    "cxpb": 0.7,                    # Crossover probability
    "tournsize": 3,                 # Tournament size
    "n_elite": 3,                   # Number of elite individuals to carry over
}