import numpy as np

def enforce_constraints(weights, min_weight, max_weight, minactive):
    """Adjust weights to satisfy portfolio constraints
    
    Args:
        weights: Array of portfolio weights
        
    Returns:
        Array of adjusted weights that satisfy all constraints
    """
    # Handle edge case: all weights are zero or nearly zero

    if np.sum(weights) < 1e-8:
        # Choose 5 random assets to have equal weights
        num_assets = len(weights)
        min_active = max(1, min(minactive, num_assets)) 
        active_indices = np.random.choice(num_assets, min_active, replace=False)
        weights = np.zeros(num_assets)
        weights[active_indices] = 1.0 / min_active
        return weights
        
    # Apply maximum weight constraint
    excess = np.maximum(0, weights - max_weight)
    weights = np.minimum(weights, max_weight)
    
    # Redistribute excess proportionally
    if np.sum(excess) > 0:
        available_indices = np.where(weights < max_weight)[0]
        if len(available_indices) > 0:
            available_weights_sum = weights[available_indices].sum()
            if available_weights_sum > 1e-10:  # Safety threshold
                weights[available_indices] += excess.sum() * weights[available_indices] / available_weights_sum
            else:
                # Handle zero-weight case: distribute equally among available indices
                weights[available_indices] += excess.sum() / len(available_indices)
    # Apply minimum weight constraint for active positions
    too_small = np.logical_and(weights > 0, weights < min_weight)
    if np.any(too_small):
        small_sum = weights[too_small].sum()
        weights[too_small] = 0
        
        # Redistribute to remaining positions
        active_indices = np.where(weights > 0)[0]
        if len(active_indices) > 0:
            weights[active_indices] += small_sum * weights[active_indices] / weights[active_indices].sum()
    
    # Critical fix: If ALL weights have been set to zero due to being too small,
    # select a few assets to have non-zero weights
    if np.sum(weights) < 1e-8:
        num_assets = len(weights)
        min_active = max(1, min(minactive, num_assets))  
        active_indices = np.random.choice(num_assets, min_active, replace=False)
        weights = np.zeros(num_assets)
        weights[active_indices] = 1.0 / min_active
        return weights
    
    # Normalize to ensure sum to 1
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        
    return weights