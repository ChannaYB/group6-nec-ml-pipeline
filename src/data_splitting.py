import pandas as pd
import numpy as np
from src.config import GROUP_COLUMN, NUM_PLANTS, RANDOM_SEED


def get_demand_groups(df):
    demand_groups = {}
    
    for demand_id in df[GROUP_COLUMN].unique():
        demand_groups[demand_id] = df[df[GROUP_COLUMN] == demand_id].copy()
    
    return demand_groups


def get_logo_splits(df, n_splits=5, random_state=RANDOM_SEED):
    # Get unique Demand IDs
    unique_demands = df[GROUP_COLUMN].unique()
    n_demands = len(unique_demands)
    
    # Shuffle demand IDs
    np.random.seed(random_state)
    shuffled_demands = np.random.permutation(unique_demands)
    
    # Split demands into k folds
    fold_size = n_demands // n_splits
    demand_folds = []
    
    for i in range(n_splits):
        start = i * fold_size
        if i == n_splits - 1:
            # Last fold gets remaining demands
            fold_demands = shuffled_demands[start:]
        else:
            fold_demands = shuffled_demands[start:start + fold_size]
        demand_folds.append(fold_demands)
    
    # Create train/val splits
    splits = []
    
    for fold_idx in range(n_splits):
        # Validation demands for this fold
        val_demands = demand_folds[fold_idx]
        
        # Training demands (all other folds)
        train_demands = np.concatenate([
            demand_folds[i] for i in range(n_splits) if i != fold_idx
        ])
        
        # Get row indices
        train_idx = df[df[GROUP_COLUMN].isin(train_demands)].index.tolist()
        val_idx = df[df[GROUP_COLUMN].isin(val_demands)].index.tolist()
        
        splits.append((train_idx, val_idx))
    
    return splits


def verify_split_integrity(train_df, test_df):
    # Check 1: No demand overlap
    train_demands = set(train_df[GROUP_COLUMN].unique())
    test_demands = set(test_df[GROUP_COLUMN].unique())
    
    overlap = train_demands & test_demands
    assert len(overlap) == 0, f"Demand ID leakage: {overlap}"
    
    # Check 2: Each demand has correct number of plants
    train_counts = train_df[GROUP_COLUMN].value_counts()
    test_counts = test_df[GROUP_COLUMN].value_counts()
    
    wrong_train = train_counts[train_counts != NUM_PLANTS]
    wrong_test = test_counts[test_counts != NUM_PLANTS]
    
    assert len(wrong_train) == 0, f"Train demands with wrong plant count: {dict(wrong_train)}"
    assert len(wrong_test) == 0, f"Test demands with wrong plant count: {dict(wrong_test)}"
    
    # Check 3: All plants present
    train_plants = set(train_df['Plant ID'].unique())
    test_plants = set(test_df['Plant ID'].unique())
    
    assert len(train_plants) == NUM_PLANTS, f"Train has {len(train_plants)} plants, expected {NUM_PLANTS}"
    assert len(test_plants) == NUM_PLANTS, f"Test has {len(test_plants)} plants, expected {NUM_PLANTS}"
    
    print(" Split integrity verified:")
    print(f"  - Train: {len(train_demands)} demands, {len(train_df)} rows")
    print(f"  - Test: {len(test_demands)} demands, {len(test_df)} rows")
    print(f"  - No demand leakage")
    print(f"  - All demands have {NUM_PLANTS} plants")
    
    return True

# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING DATA SPLITTING MODULE")
    print("="*70 + "\n")
    
    try:
        from src.data_ingestion import create_train_test
        
        # Load data
        print("Loading data...")
        train_df, test_df = create_train_test(verbose=False)
        
        # Test 1: Group data by Demand ID
        print("\nTest 1: Grouping by Demand ID")
        print("-"*70)
        groups = get_demand_groups(train_df)
        print(f" Created {len(groups)} demand groups")
        print(f"  Example: Demand 'D1' has {len(groups.get('D1', []))} rows")
        
        # Test 2: Create LOGO splits
        print("\nTest 2: Creating LOGO CV splits")
        print("-"*70)
        splits = get_logo_splits(train_df, n_splits=5)
        print(f" Created {len(splits)} CV folds")
        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"  Fold {i+1}: {len(train_idx)} train, {len(val_idx)} val")
        
        # Test 3: Verify split integrity
        print("\nTest 3: Verifying split integrity")
        print("-"*70)
        verify_split_integrity(train_df, test_df)
        
        print("\n Data Splitting Module Test PASSED!\n")
        
    except Exception as e:
        print(f"\n Error: {str(e)}\n")
        raise