"""
Unit tests: Models and Custom Scorer
Tests model training, evaluation, and LOGO CV
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_ingestion import create_train_test
from src.preprocessing import NECPreprocessor
from src.custom_scorer import (
    calculate_selection_error,
    get_selection_error_table
)
from src.models import (
    create_random_forest_model,
    create_gradient_boosting_model,
    create_linear_regression_model,
    create_model_pipeline,
    train_model,
    evaluate_model,
    evaluate_with_logo_cv
)
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


# TEST 1: Custom Scorer

def test_selection_error_calculation():
    """Test that selection error is calculated correctly"""
    print("\n[TEST 1] Selection Error Calculation")
    print("-" * 70)
    
    # Simple test case: 2 demands, 3 plants each
    demand_ids = np.array(['D1', 'D1', 'D1', 'D2', 'D2', 'D2'])
    
    # D1: Plant 0 is best (cost=10)
    # D2: Plant 1 is best (cost=20)
    y_true = np.array([10, 30, 40, 50, 20, 60])
    
    # Perfect predictions (0% error)
    y_pred_perfect = np.array([9, 31, 41, 51, 19, 61])
    error_perfect = calculate_selection_error(y_true, y_pred_perfect, demand_ids)
    assert error_perfect == 0.0, f"Expected 0% error, got {error_perfect*100}%"
    print(f"   Perfect predictions: {error_perfect*100:.1f}% error")
    
    # Wrong prediction for D2 (50% error)
    y_pred_half = np.array([9, 31, 41, 51, 61, 19])  # D2 predicts plant 2 instead of 1
    error_half = calculate_selection_error(y_true, y_pred_half, demand_ids)
    assert error_half == 0.5, f"Expected 50% error, got {error_half*100}%"
    print(f"   Half wrong: {error_half*100:.1f}% error")
    
    # All wrong (100% error)
    y_pred_wrong = np.array([31, 9, 41, 51, 61, 19])  # Both demands wrong
    error_wrong = calculate_selection_error(y_true, y_pred_wrong, demand_ids)
    assert error_wrong == 1.0, f"Expected 100% error, got {error_wrong*100}%"
    print(f"   All wrong: {error_wrong*100:.1f}% error")
    
    print(" PASSED")


def test_selection_error_table():
    """Test that selection error table is generated correctly"""
    print("\n[TEST 2] Selection Error Table Generation")
    print("-" * 70)
    
    # Create test data
    demand_ids = np.array(['D1', 'D1', 'D2', 'D2'])
    plant_ids = np.array(['P1', 'P2', 'P1', 'P2'])
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([11, 21, 31, 41])
    
    # Generate table
    table = get_selection_error_table(y_true, y_pred, demand_ids, plant_ids)
    
    # Verify table structure
    assert len(table) == 2, "Should have 2 demands"
    assert 'Demand ID' in table.columns, "Missing Demand ID column"
    assert 'True Best Plant' in table.columns, "Missing True Best Plant column"
    assert 'Predicted Best Plant' in table.columns, "Missing Predicted Best Plant column"
    assert 'Selection Correct' in table.columns, "Missing Selection Correct column"
    
    # Verify content
    assert table.iloc[0]['True Best Plant'] == 'P1', "D1 best plant should be P1"
    assert table.iloc[1]['True Best Plant'] == 'P1', "D2 best plant should be P1"
    
    print(f"   Table has {len(table)} rows")
    print(f"   All required columns present")
    print(" PASSED")


# TEST 2: Model Creation

def test_model_creation():
    """Test that all models can be created"""
    print("\n[TEST 3] Model Creation")
    print("-" * 70)
    
    # Test Random Forest
    rf_model = create_random_forest_model()
    assert rf_model is not None, "Random Forest not created"
    print("   Random Forest created")
    
    # Test Gradient Boosting
    gb_model = create_gradient_boosting_model()
    assert gb_model is not None, "Gradient Boosting not created"
    print("   Gradient Boosting created")
    
    # Test Linear Regression
    lr_model = create_linear_regression_model()
    assert lr_model is not None, "Linear Regression not created"
    print("   Linear Regression created")
    
    print(" PASSED")


def test_pipeline_creation():
    """Test that pipeline is created correctly"""
    print("\n[TEST 4] Pipeline Creation")
    print("-" * 70)
    
    # Load data to create preprocessor
    train_df, _ = create_train_test(verbose=False)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    
    # Create and fit preprocessor
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    # Test pipeline creation for each model type
    for model_type in ['random_forest', 'gradient_boosting', 'linear']:
        pipeline = create_model_pipeline(
            preprocessor_obj.preprocessor,
            model_type,
            verbose=False
        )
        
        # Verify it's a Pipeline
        assert isinstance(pipeline, Pipeline), f"{model_type} is not a Pipeline"
        
        # Verify it has preprocessor and model
        assert 'preprocessor' in pipeline.named_steps, "Missing preprocessor step"
        assert 'model' in pipeline.named_steps, "Missing model step"
        
        print(f"   {model_type.replace('_', ' ').title()} pipeline created")
    
    print(" PASSED")


# TEST 3: Model Training

def test_model_training():
    """Test that model can be trained"""
    print("\n[TEST 5] Model Training")
    print("-" * 70)
    
    # Load data
    train_df, _ = create_train_test(verbose=False)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df['Cost_USD_per_MWh']
    
    # Create preprocessor
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    # Create pipeline
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        'random_forest',
        verbose=False
    )
    
    # Train model
    trained_pipeline = train_model(pipeline, X_train, y_train, verbose=False)
    
    # Verify model is fitted
    assert hasattr(trained_pipeline, 'predict'), "Model doesn't have predict method"
    
    # Test prediction
    predictions = trained_pipeline.predict(X_train[:10])
    assert len(predictions) == 10, "Prediction length mismatch"
    assert not np.isnan(predictions).any(), "Predictions contain NaN"
    
    print(f"   Model trained successfully")
    print(f"   Predictions generated: {len(predictions)} samples")
    print(" PASSED")


# TEST 4: Model Evaluation

def test_model_evaluation():
    """Test that model evaluation works"""
    print("\n[TEST 6] Model Evaluation")
    print("-" * 70)
    
    # Load data
    train_df, test_df = create_train_test(verbose=False)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df['Cost_USD_per_MWh']
    X_test = test_df[feature_cols]
    y_test = test_df['Cost_USD_per_MWh']
    
    # Create and train model
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        'random_forest',
        verbose=False
    )
    pipeline = train_model(pipeline, X_train, y_train, verbose=False)
    
    # Evaluate
    results = evaluate_model(
        pipeline,
        X_test,
        y_test,
        test_df['Demand ID'],
        test_df['Plant ID'],
        dataset_name="Test",
        verbose=False
    )
    
    # Verify results structure
    assert 'rmse' in results, "Missing RMSE in results"
    assert 'r2' in results, "Missing R² in results"
    assert 'selection_error' in results, "Missing selection error in results"
    
    # Verify values are reasonable
    assert results['rmse'] > 0, "RMSE should be positive"
    assert 0 <= results['selection_error'] <= 1, "Selection error should be between 0 and 1"
    
    print(f"   RMSE: ${results['rmse']:.2f}/MWh")
    print(f"   R²: {results['r2']:.4f}")
    print(f"   Selection Error: {results['selection_error']:.2%}")
    print(" PASSED")


# TEST 5: LOGO Cross-Validation

def test_logo_cv():
    """Test that LOGO CV works correctly"""
    print("\n[TEST 7] LOGO Cross-Validation")
    print("-" * 70)
    
    # Load data
    train_df, _ = create_train_test(verbose=False)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    
    # Create preprocessor
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    # Create pipeline (unfitted)
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        'linear',  # Use Linear for speed
        verbose=False
    )
    
    # Run LOGO CV (3 folds for speed)
    cv_results = evaluate_with_logo_cv(
        pipeline,
        train_df,
        n_splits=3,
        verbose=False
    )
    
    # Verify CV results
    assert cv_results['n_splits'] == 3, "Wrong number of splits"
    assert len(cv_results['fold_results']) == 3, "Wrong number of fold results"
    assert 'avg_rmse' in cv_results, "Missing average RMSE"
    assert 'avg_selection_error' in cv_results, "Missing average selection error"
    
    # Verify all folds have results
    for fold_result in cv_results['fold_results']:
        assert 'rmse' in fold_result, "Missing RMSE in fold"
        assert 'selection_error' in fold_result, "Missing selection error in fold"
    
    print(f"   {cv_results['n_splits']} folds completed")
    print(f"   Average RMSE: ${cv_results['avg_rmse']:.2f}")
    print(f"   Average Selection Error: {cv_results['avg_selection_error']:.2%}")
    print(" PASSED")


# TEST 6: No Data Leakage
def test_no_data_leakage():
    """Test that there's no data leakage in LOGO CV"""
    print("\n[TEST 8] No Data Leakage in LOGO CV")
    print("-" * 70)
    
    # Load data
    train_df, _ = create_train_test(verbose=False)
    
    # Get LOGO splits
    from src.data_splitting import get_logo_splits
    splits = get_logo_splits(train_df, n_splits=3)
    
    # Check each fold
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Get demand IDs
        train_demands = set(train_df.iloc[train_idx]['Demand ID'].unique())
        val_demands = set(train_df.iloc[val_idx]['Demand ID'].unique())
        
        # Check no overlap
        overlap = train_demands & val_demands
        assert len(overlap) == 0, f"Fold {fold_idx+1}: Data leakage detected! {len(overlap)} demands in both train and val"
        
        print(f"   Fold {fold_idx+1}: {len(train_demands)} train demands, {len(val_demands)} val demands, no overlap")
    
    print(" PASSED")


# TEST 7: Integration Test

def test_full_pipeline_integration():
    """Test complete pipeline from data loading to evaluation"""
    print("\n[TEST 9] Full Pipeline Integration")
    print("-" * 70)
    
    # 1. Load data
    train_df, test_df = create_train_test(verbose=False)
    print("   Data loaded")
    
    # 2. Preprocess
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df['Cost_USD_per_MWh']
    X_test = test_df[feature_cols]
    y_test = test_df['Cost_USD_per_MWh']
    
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    print("   Preprocessing fitted")
    
    # 3. Create and train model
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        'random_forest',
        verbose=False
    )
    pipeline = train_model(pipeline, X_train, y_train, verbose=False)
    print("   Model trained")
    
    # 4. Evaluate
    results = evaluate_model(
        pipeline,
        X_test,
        y_test,
        test_df['Demand ID'],
        test_df['Plant ID'],
        dataset_name="Test",
        verbose=False
    )
    print("   Model evaluated")
    
    # Verify end-to-end works
    assert results['rmse'] > 0, "Integration failed: invalid RMSE"
    assert 0 <= results['selection_error'] <= 1, "Integration failed: invalid selection error"
    
    print(f"   Integration successful: RMSE=${results['rmse']:.2f}, Error={results['selection_error']:.2%}")
    print(" PASSED")


# RUN ALL TESTS

if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIT TESTS")
    print("="*70)
    
    tests = [
        test_selection_error_calculation,
        test_selection_error_table,
        test_model_creation,
        test_pipeline_creation,
        test_model_training,
        test_model_evaluation,
        test_logo_cv,
        test_no_data_leakage,
        test_full_pipeline_integration
    ]
    
    failed_tests = []
    
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f" FAILED: {str(e)}")
            failed_tests.append(test_func.__name__)
        except Exception as e:
            print(f" ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_tests.append(test_func.__name__)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {len(tests) - len(failed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test_name in failed_tests:
            print(f"   {test_name}")
        print("\n" + "="*70)
        print(" SOME TESTS FAILED")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print(" ALL TESTS PASSED!")
        print("="*70 + "\n")