import numpy as np
import random
from deap import base, creator, tools, algorithms
from portfolio_evaluation import portfolio_performance
from constraint_enforce import enforce_constraints
from data_config import ga_params, max_seconds, time_to_convergence, min_weight, max_weight, MINACTIVE
import time
import helper as hp

def deap_portfolio_optimization(expected_returns, covariance_matrix, population_size:int=ga_params["population_size"], mutation_mutpb:float=ga_params["mutation_mutpb"], mutation_indpb:float=ga_params["mutation_indpb"], cxpb:float=ga_params["cxpb"], tournsize:float=ga_params["tournsize"], n_elite:int=ga_params["n_elite"], stop_on_convergence:bool=False, min_weight:float=min_weight, max_weight:float=max_weight, minactive:int=MINACTIVE):
    """Portfolio optimization using DEAP's genetic algorithm with convergence checking
    
    Args:
        expected_returns: Array or dict of expected returns
        covariance_matrix: Covariance matrix
            
    Returns:
        Tuple of (best_weights, best_sharpe, generations_completed)
    """
    start_time = time.time()

    expected_returns_tickers, expected_returns_array = hp.prepare_return_lists(expected_returns)

    
    num_assets = len(expected_returns_array)
    
    # Need to clear any previous DEAP definitions
    if 'FitnessMax' in creator.__dict__:
        del creator.FitnessMax
    if 'Individual' in creator.__dict__:
        del creator.Individual
    
    # Define a maximization problem (for Sharpe ratio)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Individual initialization
    def create_individual():
        weights = np.random.dirichlet(np.ones(num_assets))
        weights = enforce_constraints(weights, min_weight, max_weight, minactive)
        return list(weights)
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluation function
    def evaluate(individual):
        weights = np.array(individual)
        weights = enforce_constraints(weights, min_weight=min_weight, max_weight=max_weight, minactive=minactive)
        
        # Update individual with enforced weights
        for i in range(len(individual)):
            individual[i] = weights[i]
            
        sharpe, _, _ = portfolio_performance(weights, expected_returns_array, covariance_matrix)
        
        if not np.isfinite(sharpe):
            return (-1000.0,)
        return (sharpe,)
    
    # Custom crossover
    def custom_crossover(ind1, ind2):
        child1, child2 = list(ind1), list(ind2)
        
        # Two-point crossover
        if len(ind1) > 2:
            points = sorted(random.sample(range(num_assets), 2))
            for i in range(points[0], points[1]):
                child1[i], child2[i] = child2[i], child1[i]
        else:
            # With fewer assets, do simple swap
            point = random.randint(0, len(ind1)-1)
            child1[point], child2[point] = child2[point], child1[point]
            
        # Normalize
        child1 = np.array(child1)
        child2 = np.array(child2)
        
        if np.sum(child1) > 0:
            child1 = child1 / np.sum(child1)
        if np.sum(child2) > 0:
            child2 = child2 / np.sum(child2)
            
        # Apply constraints
        child1 = enforce_constraints(child1, min_weight, max_weight, minactive)
        child2 = enforce_constraints(child2, min_weight, max_weight, minactive)
        
        return creator.Individual(child1), creator.Individual(child2)
    
    def custom_mutation(individual, indpb):
        weights = np.array(individual)
        
        for i in range(len(weights)):
            if random.random() < indpb:
                change = random.uniform(-0.1, 0.1) * weights[i]
                weights[i] += change
        
        weights = np.maximum(0, weights)
        
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            
        weights = enforce_constraints(weights, min_weight, max_weight, minactive)
        
        for i in range(len(individual)):
            individual[i] = weights[i]
        
        return (individual,)
    
    # Register GA operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, indpb=mutation_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    
    # Create initial population
    pop = toolbox.population(n=population_size)
    
    # Hall of fame to keep track of the best individual
    hof = tools.HallOfFame(n_elite)
    
    # History for convergence checking
    history = []
    history_weights = []
    timeline = []
    
    # Run the algorithm with convergence checking
    generation = 0
    last_increase_time = time.time()
    while True:
        generation += 1
        
        # Evaluate the current population if not already done
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update hall of fame with current population
        hof.update(pop)
        
        # Create offspring through selection, crossover and mutation
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutation_mutpb)
        
        # Evaluate the offspring
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # Elitist selection: combine elite individuals with offspring
        elite_individuals = [toolbox.clone(ind) for ind in hof]
        
        # Select the next generation from offspring + elite
        combined_pop = offspring + elite_individuals
        pop = toolbox.select(combined_pop, k=population_size)

        curr_time = time.time()

        if curr_time - start_time >= max_seconds:
            num_generations = generation + 1
            break

        if stop_on_convergence and curr_time - last_increase_time > time_to_convergence:
            # Check if we have not improved for 5 seconds
            num_generations = generation + 1
            break

        # Get current best fitness
        if hof:
            best_fitness = hof[0].fitness.values[0]

            if len(history) == 0 or not np.isclose(history[-1], best_fitness, atol=1e-6):
                history.append(best_fitness)
                timeline.append(curr_time-start_time)
                history_weights.append(hp.create_best_weights_dict(expected_returns_tickers, np.array(hof[0])))
                last_increase_time = curr_time
    
    # Get best solution after all generations
    best_individual = hof[0]
    best_weights = np.array(best_individual)
    best_weights = enforce_constraints(best_weights, min_weight, max_weight, minactive)
    best_sharpe = evaluate([w for w in best_weights])[0]

    # extrapolate the last point if needed
    if len(timeline) == 0 or not np.isclose(timeline[-1], max_seconds, atol=1e-6):
        timeline.append(max_seconds)
        history.append(best_sharpe)
        history_weights.append(hp.create_best_weights_dict(expected_returns_tickers, best_weights))
    
    return best_weights, best_sharpe, num_generations, history,history_weights, timeline