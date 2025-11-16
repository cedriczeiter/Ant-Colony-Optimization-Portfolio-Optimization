# Ant Colony Optimization for Portfolio Optimization

A comparative study of metaheuristic algorithms for solving the portfolio optimization problem, developed as part of the course **227-0707-00 S Optimization Methods for Engineers** at ETH Zürich (Spring 2025).

**Authors:** Cédric Zeiter & Alexandre Faroux

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
├── summarized_results.txt         # Final submission, containing various plots
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
git clone git@github.com:cedriczeiter/Ant-Colony-Optimization-Portfolio-Optimization.git
cd Ant-Colony-Optimization-Portfolio-Optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
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

## Acknowledgments

- **ETH Zürich** - For providing the course framework
- **DEAP Team** - For the excellent genetic algorithm library
- **Yahoo Finance** - For accessible historical stock data


---

**Course**: 227-0707-00 S Optimization Methods for Engineers  
**Institution**: ETH Zürich  
**Semester**: Spring 2025  
**Authors**: Cédric Zeiter & Alexandre Faroux
