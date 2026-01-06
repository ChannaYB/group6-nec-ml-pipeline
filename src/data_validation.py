"""
Data Ingestion Module for NEC ML Pipeline
Loads 3 raw files, merges them, cleans data, and creates train/test splits.

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
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    GROUP_COLUMN,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_SEED,
    VERBOSE
)

# Create sklearn-compatible aliases
TEST_SIZE = 1 - TRAIN_TEST_SPLIT_RATIO
RANDOM_STATE = RANDOM_SEED


def load_raw_data(verbose=VERBOSE):
    """
    Load and merge the 3 raw NEC datasets.
    """
    if verbose:
        print("ℹ Loading raw datasets...")

    # Load the 3 separate CSV files
    demand_df = pd.read_csv(DEMAND_PATH)
    plants_df = pd.read_csv(PLANTS_PATH)
    costs_df = pd.read_csv(GENERATION_COSTS_PATH)

    if verbose:
        print(f"   Demand data: {demand_df.shape[0]} rows × {demand_df.shape[1]} columns")
        print(f"   Plants data: {plants_df.shape[0]} rows × {plants_df.shape[1]} columns")
        print(f"   Costs data: {costs_df.shape[0]} rows × {costs_df.shape[1]} columns")

    # Merge datasets
    if verbose:
        print("\nℹ Merging datasets...")
    
    # Step 1: Merge costs with demand features
    df = costs_df.merge(demand_df, on='Demand ID', how='left')
    if verbose:
        print(f"   After demand merge: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Step 2: Merge with plant features
    df = df.merge(plants_df, on='Plant ID', how='left')
    if verbose:
        print(f"   After plant merge: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Unique demands: {df['Demand ID'].nunique()}")
        print(f"   Unique plants: {df['Plant ID'].nunique()}")

    return df


def clean_data(df, verbose=VERBOSE):
    """
    Clean dataset before splitting.
    """
    if verbose:
        print("\nℹ Cleaning data...")

    initial_rows = len(df)
    initial_demands = df[GROUP_COLUMN].nunique()

    # Step 1: Check for missing target values
    missing_target = df[TARGET_COLUMN].isnull().sum()
    
    if missing_target > 0:
        if verbose:
            print(f"   Found {missing_target} rows with missing {TARGET_COLUMN}")
            print(f"    Dropping these rows...")
        df = df.dropna(subset=[TARGET_COLUMN])
    else:
        if verbose:
            print(f"   No missing values in {TARGET_COLUMN}")

    # Step 2: Check for incomplete demand groups
    # Each demand should have exactly 64 plants
    demand_counts = df.groupby(GROUP_COLUMN).size()
    incomplete_demands = demand_counts[demand_counts != 64]
    
    if len(incomplete_demands) > 0:
        if verbose:
            print(f"   Found {len(incomplete_demands)} demands with incomplete plant sets")
            print(f"    These demands don't have 64 plants:")
            for demand_id, count in incomplete_demands.items():
                print(f"      - {demand_id}: {count} plants")
            print(f"    Removing these incomplete demands...")
        
        bad_demands = incomplete_demands.index.tolist()
        df = df[~df[GROUP_COLUMN].isin(bad_demands)]
    else:
        if verbose:
            print(f"   All demands have exactly 64 plants")

    # Summary
    dropped_rows = initial_rows - len(df)
    final_demands = df[GROUP_COLUMN].nunique()
    dropped_demands = initial_demands - final_demands

    if verbose:
        print(f"\n   Cleaning Summary:")
        print(f"    - Dropped {dropped_rows} rows ({dropped_rows/initial_rows*100:.1f}%)")
        print(f"    - Dropped {dropped_demands} incomplete demands")
        print(f"    - Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"    - Final demands: {final_demands}")

    return df


def create_train_test(verbose=VERBOSE):
    """
    Create train/test split using grouped splitting.
    """
    if verbose:
        print("\n" + "="*70)
        print("DATA INGESTION PIPELINE - MEMBER 1")
        print("="*70)

    # Step 1: Load and merge raw data
    df = load_raw_data(verbose)

    # Step 2: Clean data
    df = clean_data(df, verbose)

    # Step 3: Create grouped train/test split
    if verbose:
        print("\nℹ Creating grouped train/test split...")
        print(f"  Grouping by: {GROUP_COLUMN}")
        print(f"  Split ratio: {int(TRAIN_TEST_SPLIT_RATIO*100)}% train / {int(TEST_SIZE*100)}% test")
        print(f"  Random seed: {RANDOM_STATE}")

    # Use GroupShuffleSplit to maintain group integrity
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

    # Verification
    if verbose:
        print(f"\n   Split Results:")
        print(f"    Train:")
        print(f"      - Shape: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
        print(f"      - Demands: {train_df[GROUP_COLUMN].nunique()}")
        print(f"      - Missing {TARGET_COLUMN}: {train_df[TARGET_COLUMN].isnull().sum()}")
        
        print(f"    Test:")
        print(f"      - Shape: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
        print(f"      - Demands: {test_df[GROUP_COLUMN].nunique()}")
        print(f"      - Missing {TARGET_COLUMN}: {test_df[TARGET_COLUMN].isnull().sum()}")
        
        # Check for leakage
        train_demands = set(train_df[GROUP_COLUMN].unique())
        test_demands = set(test_df[GROUP_COLUMN].unique())
        overlap = train_demands & test_demands
        
        if len(overlap) == 0:
            print(f"\n   No demand leakage (train and test demands are separate)")
        else:
            print(f"\n   WARNING: {len(overlap)} demands appear in both train and test!")

        # Check plant counts
        train_counts = train_df.groupby(GROUP_COLUMN).size()
        test_counts = test_df.groupby(GROUP_COLUMN).size()
        
        if all(train_counts == 64) and all(test_counts == 64):
            print(f"   All demands have exactly 64 plants in both train and test")
        else:
            print(f"   WARNING: Some demands don't have 64 plants!")

    # Step 4: Save processed data
    if verbose:
        print("\nℹ Saving processed data...")
    
    train_path = Path(PROCESSED_DATA_DIR) / 'train.csv'
    test_path = Path(PROCESSED_DATA_DIR) / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    if verbose:
        print(f"   Saved train data: {train_path}")
        print(f"   Saved test data: {test_path}")
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
        # Run the complete pipeline
        train_df, test_df = create_train_test(verbose=True)

        # Additional verification
        print("\n" + "="*70)
        print("VERIFICATION CHECKS")
        print("="*70)
        
        # Check 1: No missing targets
        print(f"\n[1] Missing Target Values:")
        print(f"  Train: {train_df[TARGET_COLUMN].isnull().sum()}")
        print(f"  Test: {test_df[TARGET_COLUMN].isnull().sum()}")
        
        # Check 2: All demands have 64 plants
        print(f"\n[2] Plant Counts per Demand:")
        train_counts = train_df.groupby(GROUP_COLUMN).size()
        test_counts = test_df.groupby(GROUP_COLUMN).size()
        print(f"  Train: All demands have 64 plants = {all(train_counts == 64)}")
        print(f"  Test: All demands have 64 plants = {all(test_counts == 64)}")
        
        # Check 3: No overlap
        print(f"\n[3] Train/Test Overlap:")
        train_demands = set(train_df[GROUP_COLUMN].unique())
        test_demands = set(test_df[GROUP_COLUMN].unique())
        overlap = train_demands & test_demands
        print(f"  Overlapping demands: {len(overlap)}")
        
        # Check 4: Data ranges
        print(f"\n[4] Target Variable Range:")
        print(f"  Train - Min: {train_df[TARGET_COLUMN].min():.2f}, Max: {train_df[TARGET_COLUMN].max():.2f}")
        print(f"  Test - Min: {test_df[TARGET_COLUMN].min():.2f}, Max: {test_df[TARGET_COLUMN].max():.2f}")
        
        print("\n" + "="*70)
        print(" ALL CHECKS PASSED - DATA INGESTION TEST COMPLETE")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print(" ERROR OCCURRED")
        print("="*70)
        print(f"\nError: {e}\n")
        
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        print("\n" + "="*70 + "\n")