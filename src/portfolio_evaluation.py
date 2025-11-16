import numpy as np
from data_config import risk_free_rate
import math

def portfolio_performance(weights, expected_returns, cov_matrix):
    """Calculate portfolio performance metrics
    
    Args:
        weights: Array of portfolio weights
        expected_returns: Array of annualized expected returns for each asset (decimal)
        cov_matrix: Covariance matrix of returns (annulaized)
        
    Returns:
        Tuple of (sharpe ratio, portfolio return, portfolio volatility)
    """
    # Handle if weights are None
    if weights is None:
        return float('-inf'), 0, float('inf')
    
    assert len(weights) == len(expected_returns), "Weights and expected returns must match in length"
    assert cov_matrix.shape[0] == cov_matrix.shape[1], "Covariance matrix must be square"
    assert math.isclose(np.sum(weights), 1.0), f"Weights must sum to 1, it's currently {np.sum(weights)}"
        
    port_return = np.dot(weights, expected_returns)
    
    # Calculate volatility with numerical stability check
    try:
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Ensure we don't divide by zero
        if port_vol < 1e-8:  # Small threshold to prevent near-zero division
            port_vol = 1e-8
        sharpe = (port_return - risk_free_rate) / port_vol
    except:
        # In case of numerical errors
        sharpe = float('-inf')
        port_vol = float('inf')
    
    return sharpe, port_return, port_vol