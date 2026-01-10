"""
Custom Scorer for NEC ML Pipeline
Implements selection error rate for plant selection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer

from src.config import GROUP_COLUMN, TARGET_COLUMN, PLANTS_PER_DEMAND


def calculate_selection_error(y_true, y_pred, demand_ids):
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    demand_ids = np.array(demand_ids)
    
    # Get unique demands
    unique_demands = np.unique(demand_ids)
    
    errors = 0
    total_demands = len(unique_demands)
    
    for demand_id in unique_demands:
        # Get indices for this demand
        mask = demand_ids == demand_id
        
        # Get true and predicted costs for this demand
        true_costs = y_true[mask]
        pred_costs = y_pred[mask]
        
        # Find best plant (minimum cost)
        true_best_idx = np.argmin(true_costs)
        pred_best_idx = np.argmin(pred_costs)
        
        # Check if same plant selected
        if true_best_idx != pred_best_idx:
            errors += 1
    
    # Calculate error rate
    selection_error_rate = errors / total_demands
    
    return selection_error_rate


def selection_error_scorer(estimator, X, y, demand_ids=None):
    """
    Calculate plant selection error rate.
    """
    if demand_ids is None:
        raise ValueError("demand_ids must be provided for selection error calculation")
    
    # Get predictions
    y_pred = estimator.predict(X)
    
    # Calculate selection error
    error = calculate_selection_error(y, y_pred, demand_ids)
    
    # Return negative (sklearn convention: higher score is better)
    return -error


def get_selection_error_table(y_true, y_pred, demand_ids, plant_ids):
    """
    Generate detailed selection error table.
    """
    
    # Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    demand_ids = np.array(demand_ids)
    plant_ids = np.array(plant_ids)
    
    results = []
    
    for demand_id in np.unique(demand_ids):
        # Get data for this demand
        mask = demand_ids == demand_id
        
        true_costs = y_true[mask]
        pred_costs = y_pred[mask]
        plants = plant_ids[mask]
        
        # Find best plants
        true_best_idx = np.argmin(true_costs)
        pred_best_idx = np.argmin(pred_costs)
        
        true_best_plant = plants[true_best_idx]
        pred_best_plant = plants[pred_best_idx]
        
        # Check if correct
        is_correct = (true_best_plant == pred_best_plant)
        
        results.append({
            'Demand ID': demand_id,
            'True Best Plant': true_best_plant,
            'Predicted Best Plant': pred_best_plant,
            'Selection Correct': is_correct,
            'True Best Cost': true_costs[true_best_idx],
            'Predicted Cost': pred_costs[pred_best_idx],
            'Cost Difference': pred_costs[pred_best_idx] - true_costs[true_best_idx]
        })
    
    return pd.DataFrame(results)

# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING CUSTOM SCORER")
    print("="*70)
    
    # Test with synthetic data
    print("\n[Test 1] Basic selection error calculation")
    print("-"*70)
    
    # Simulate 3 demands, each with 4 plants
    demand_ids = np.array(['D1']*4 + ['D2']*4 + ['D3']*4)
    plant_ids = np.array(['P1', 'P2', 'P3', 'P4']*3)
    
    # True costs (P1 is best for D1, P2 for D2, P3 for D3)
    y_true = np.array([
        10, 20, 30, 40,  # D1: P1 is best
        40, 10, 30, 20,  # D2: P2 is best
        30, 40, 10, 20   # D3: P3 is best
    ])
    
    # Predicted costs (correct for D1 and D2, wrong for D3)
    y_pred = np.array([
        9, 21, 31, 41,   # D1: Predicts P1 (correct)
        41, 9, 31, 21,   # D2: Predicts P2 (correct)
        31, 41, 11, 8    # D3: Predicts P4 (WRONG - should be P3)
    ])
    
    error_rate = calculate_selection_error(y_true, y_pred, demand_ids)
    
    print(f"Selection Error Rate: {error_rate:.2%}")
    print(f"Expected: 33.33% (1 out of 3 wrong)")
    
    assert abs(error_rate - 1/3) < 0.01, "Selection error calculation failed"
    print(" Test passed")
    
    # Test with selection table
    print("\n[Test 2] Selection error table")
    print("-"*70)
    
    table = get_selection_error_table(y_true, y_pred, demand_ids, plant_ids)
    print("\n", table.to_string(index=False))
    
    print("\n Custom scorer tests passed\n")