"""
Unit tests for data ingestion module
"""

import pytest
import pandas as pd
import os
from src.data_ingestion import DataIngestion, load_and_merge_data
from src.config import NUM_DEMAND_SCENARIOS, NUM_PLANTS


def test_load_raw_data():
    """Test loading of raw CSV files."""
    ingestion = DataIngestion(verbose=False)
    demand_df, plants_df, costs_df = ingestion.load_raw_data()
    
    # Check DataFrames are loaded
    assert demand_df is not None
    assert plants_df is not None
    assert costs_df is not None
    
    # Check shapes
    assert len(demand_df) == NUM_DEMAND_SCENARIOS
    assert len(plants_df) == NUM_PLANTS
    
    print(" test_load_raw_data PASSED")


def test_merge_datasets():
    """Test merging of three datasets."""
    ingestion = DataIngestion(verbose=False)
    ingestion.load_raw_data()
    merged_df = ingestion.merge_datasets()
    
    # Check merged dataset
    assert merged_df is not None
    assert len(merged_df) == NUM_DEMAND_SCENARIOS * NUM_PLANTS
    
    # Check key columns exist
    assert 'Demand ID' in merged_df.columns
    assert 'Plant ID' in merged_df.columns
    assert 'Cost_USD_per_MWh' in merged_df.columns
    
    print(" test_merge_datasets PASSED")


def test_train_test_split():
    """Test train/test splitting by Demand ID."""
    ingestion = DataIngestion(verbose=False)
    ingestion.load_raw_data()
    ingestion.merge_datasets()
    train_df, test_df = ingestion.create_train_test_split()
    
    # Check split exists
    assert train_df is not None
    assert test_df is not None
    
    # Check no overlap in Demand IDs
    train_demands = set(train_df['Demand ID'].unique())
    test_demands = set(test_df['Demand ID'].unique())
    assert len(train_demands & test_demands) == 0
    
    # Check each demand has 64 plants
    train_counts = train_df['Demand ID'].value_counts()
    test_counts = test_df['Demand ID'].value_counts()
    
    assert all(train_counts == NUM_PLANTS)
    assert all(test_counts == NUM_PLANTS)
    
    print(" test_train_test_split PASSED")


if __name__ == "__main__":
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS FOR DATA INGESTION")
    print("="*70 + "\n")
    
    try:
        test_load_raw_data()
        test_merge_datasets()
        test_train_test_split()
        
        print("\n" + "="*70)
        print(" ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n Test failed: {str(e)}\n")
        raise
    except Exception as e:
        print(f"\n Error: {str(e)}\n")
        raise