"""
Test Data Validation Module
Run from main folder: python test_validation.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("TESTING DATA VALIDATION MODULE")
print("="*70)

from src.data_ingestion import create_train_test
from src.data_validation import validate_data

# Load data
print("\nLoading data...")
train_df, test_df = create_train_test(verbose=False)
print(f" Data loaded: Train {train_df.shape}, Test {test_df.shape}")

# Run validation
print("\nRunning validation...")
passed = validate_data(train_df, test_df, verbose=True)

# Summary
print("\n" + "="*70)
if passed:
    print(" VALIDATION TEST PASSED!")
else:
    print(" VALIDATION TEST COMPLETED WITH WARNINGS")
print("="*70)