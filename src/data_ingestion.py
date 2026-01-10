"""
Data Ingestion Module for NEC ML Pipeline
----------------------------------------
Loads raw data, cleans it, and creates train/test splits.

Author:
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    DEMAND_PATH,
    PLANTS_PATH,
    GENERATION_COSTS_PATH,
    TARGET_COLUMN,
    GROUP_COLUMN,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_SEED,
    VERBOSE
)

# Create sklearn-compatible aliases
TEST_SIZE = 1 - TRAIN_TEST_SPLIT_RATIO  # 0.2
RANDOM_STATE = RANDOM_SEED              # 42


def load_raw_data(verbose=VERBOSE):
    """Load and merge the 3 raw datasets."""
    if verbose:
        print("ℹ Loading raw datasets...")

    # Load the 3 separate files
    demand_df = pd.read_csv(DEMAND_PATH)
    plants_df = pd.read_csv(PLANTS_PATH)
    costs_df = pd.read_csv(GENERATION_COSTS_PATH)

    if verbose:
        print(f"   Demand data: {demand_df.shape}")
        print(f"   Plants data: {plants_df.shape}")
        print(f"   Costs data: {costs_df.shape}")

    # Merge datasets
    if verbose:
        print("\nℹ Merging datasets...")
    
    df = costs_df.merge(demand_df, on='Demand ID', how='left')
    df = df.merge(plants_df, on='Plant ID', how='left')

    if verbose:
        print(f"   Merged data: {df.shape}")

    return df


def clean_data(df, verbose=VERBOSE):
    """
    Clean dataset before splitting.
    - Drop rows with missing target (CRITICAL)
    - Remove incomplete demand groups (don't have 64 plants)
    """
    if verbose:
        print("\nℹ Cleaning data...")

    initial_rows = len(df)

    # Step 1: Drop rows with missing target
    missing_target = df[TARGET_COLUMN].isnull().sum()
    if missing_target > 0:
        if verbose:
            print(f"   Found {missing_target} rows with missing target")
        df = df.dropna(subset=[TARGET_COLUMN])

    # Step 2: Check for incomplete demand groups
    demand_counts = df.groupby(GROUP_COLUMN).size()
    incomplete_demands = demand_counts[demand_counts != 64]
    
    if len(incomplete_demands) > 0:
        if verbose:
            print(f"   Found {len(incomplete_demands)} demands with incomplete plant sets")
            print(f"     Removing these demands: {incomplete_demands.index.tolist()}")
        
        bad_demands = incomplete_demands.index.tolist()
        df = df[~df[GROUP_COLUMN].isin(bad_demands)]

    dropped = initial_rows - len(df)

    if verbose:
        print(f"   Dropped {dropped} rows total")
        print(f"   Cleaned data shape: {df.shape}")
        print(f"   Unique demands: {df[GROUP_COLUMN].nunique()}")

    return df


def create_train_test(verbose=VERBOSE):
    """
    Create train/test split using LOGO-style grouping.
    Ensures no demand leakage between train and test.
    """
    if verbose:
        print("\n" + "="*70)
        print("DATA INGESTION - MEMBER 1")
        print("="*70)

    # Load + clean data
    df = load_raw_data(verbose)
    df = clean_data(df, verbose)

    # Group-based split (no leakage)
    if verbose:
        print("\nℹ Creating grouped train/test split...")
        print(f"  Using {GROUP_COLUMN} as grouping variable")
        print(f"  Split ratio: {int(TRAIN_TEST_SPLIT_RATIO*100)}% train / {int(TEST_SIZE*100)}% test")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df[GROUP_COLUMN])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    if verbose:
        print(f"\n   Train shape: {train_df.shape}")
        print(f"    - Train demands: {train_df[GROUP_COLUMN].nunique()}")
        print(f"   Test shape: {test_df.shape}")
        print(f"    - Test demands: {test_df[GROUP_COLUMN].nunique()}")
        
        # Verify no leakage
        train_demands = set(train_df[GROUP_COLUMN].unique())
        test_demands = set(test_df[GROUP_COLUMN].unique())
        overlap = train_demands & test_demands
        
        if len(overlap) == 0:
            print(f"   No demand leakage verified")
        else:
            print(f"   WARNING: {len(overlap)} demands in both train and test!")

    # Save to processed folder
    if verbose:
        print("\nℹ Saving processed data...")
    
    from src.config import PROCESSED_DATA_DIR
    train_path = Path(PROCESSED_DATA_DIR) / 'train.csv'
    test_path = Path(PROCESSED_DATA_DIR) / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    if verbose:
        print(f"   Saved: {train_path}")
        print(f"   Saved: {test_path}")
        print("\n" + "="*70)
        print(" DATA INGESTION COMPLETE")
        print("="*70 + "\n")

    return train_df, test_df


# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING DATA INGESTION MODULE")
    print("="*70)

    try:
        train_df, test_df = create_train_test(verbose=True)

        # Quick verification
        print("\n" + "-"*70)
        print("VERIFICATION")
        print("-"*70)
        print(f"Train - Missing target: {train_df[TARGET_COLUMN].isnull().sum()}")
        print(f"Test - Missing target: {test_df[TARGET_COLUMN].isnull().sum()}")
        
        train_counts = train_df.groupby(GROUP_COLUMN).size()
        test_counts = test_df.groupby(GROUP_COLUMN).size()
        
        print(f"Train - All demands have 64 plants: {all(train_counts == 64)}")
        print(f"Test - All demands have 64 plants: {all(test_counts == 64)}")
        
        print("\n Data ingestion test completed successfully\n")

    except Exception as e:
        print(f"\n Error: {e}\n")
        import traceback
        traceback.print_exc()
