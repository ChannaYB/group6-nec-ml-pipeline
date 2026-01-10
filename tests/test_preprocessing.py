"""
Unit tests for preprocessing module
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data_ingestion import create_train_test
from src.preprocessing import preprocess_data, NECPreprocessor

def test_preprocessing_pipeline():
    """Test complete preprocessing pipeline"""
    print("\n[TEST 1] Complete preprocessing pipeline")
    
    # Load data
    train_df, test_df = create_train_test(verbose=False)
    
    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        train_df, test_df, verbose=False
    )
    
    # Assertions
    assert X_train.shape[0] == len(train_df), "Train rows mismatch"
    assert X_test.shape[0] == len(test_df), "Test rows mismatch"
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"
    assert np.isnan(X_train).sum() == 0, "NaN in train data"
    assert np.isnan(X_test).sum() == 0, "NaN in test data"
    
    print(" PASSED")

def test_no_data_leakage():
    """Test preprocessor doesn't leak test data"""
    print("\n[TEST 2] No data leakage")
    
    train_df, test_df = create_train_test(verbose=False)
    
    from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Fit on train only
    preprocessor = NECPreprocessor(verbose=False)
    preprocessor.fit(X_train)
    
    # Transform both
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    
    # Should have same features
    assert X_train_t.shape[1] == X_test_t.shape[1], "Feature mismatch"
    
    print(" PASSED")

def test_feature_count():
    """Test output feature count is correct"""
    print("\n[TEST 3] Feature count")
    
    train_df, test_df = create_train_test(verbose=False)
    X_train, X_test, _, _, _ = preprocess_data(train_df, test_df, verbose=False)
    
    # Should have 41 features (30 numerical + 11 encoded categorical)
    assert X_train.shape[1] == 41, f"Expected 41 features, got {X_train.shape[1]}"
    
    print(" PASSED")

if __name__ == "__main__":
    print("="*70)
    print("RUNNING PREPROCESSING UNIT TESTS")
    print("="*70)
    
    try:
        test_preprocessing_pipeline()
        test_no_data_leakage()
        test_feature_count()
        
        print("\n" + "="*70)
        print(" ALL PREPROCESSING TESTS PASSED")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n Test failed: {e}\n")
        raise