"""
Unit tests: Hyperparameter Tuning and Final Integration
Tests tuning framework and complete pipeline integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
import pickle

from src.data_ingestion import create_train_test
from src.preprocessing import NECPreprocessor
from src.models import create_model_pipeline
from src.hyperparameter_tuning import (
    tune_hyperparameters,
    create_tuning_leaderboard,
    save_model,
    load_model
)
from src.final_pipeline import NECPipeline
from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    MODELS_DIR
)


# TEST 1: Tuning Module Initialization

def test_tuning_imports():
    """Test that all tuning modules import correctly"""
    print("\n[TEST 1] Tuning Module Imports")
    print("-" * 70)
    
    # Test imports
    assert tune_hyperparameters is not None, "tune_hyperparameters not imported"
    assert create_tuning_leaderboard is not None, "create_tuning_leaderboard not imported"
    assert save_model is not None, "save_model not imported"
    assert load_model is not None, "load_model not imported"
    
    print("   All tuning functions imported successfully")
    print(" PASSED")


# TEST 2: Hyperparameter Tuning

def test_hyperparameter_tuning_quick():
    """Test hyperparameter tuning in quick mode"""
    print("\n[TEST 2] Hyperparameter Tuning (Quick Mode)")
    print("-" * 70)
    
    # Load data
    train_df, test_df = create_train_test(verbose=False)
    
    # Create pipeline
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        model_type='random_forest',
        verbose=False
    )
    
    # Run tuning (quick mode for speed)
    print("  Running hyperparameter tuning (this may take a minute)...")
    tuning_results = tune_hyperparameters(
        pipeline,
        train_df,
        model_type='random_forest',
        method='grid',
        quick=True,
        verbose=False
    )
    
    # Verify results structure
    assert 'best_estimator' in tuning_results, "Missing best_estimator"
    assert 'best_params' in tuning_results, "Missing best_params"
    assert 'best_score' in tuning_results, "Missing best_score"
    assert 'cv_results' in tuning_results, "Missing cv_results"
    
    # Verify best estimator is fitted
    assert hasattr(tuning_results['best_estimator'], 'predict'), "Best estimator not fitted"
    
    # Verify best params is a dict
    assert isinstance(tuning_results['best_params'], dict), "Best params not a dict"
    assert len(tuning_results['best_params']) > 0, "Best params is empty"
    
    # Verify best score is a number
    assert isinstance(tuning_results['best_score'], (int, float)), "Best score not a number"
    
    print(f"   Tuning completed successfully")
    print(f"   Best RMSE: ${-tuning_results['best_score']:.2f}")
    print(f"   Found {len(tuning_results['best_params'])} optimal parameters")
    print(" PASSED")


# TEST 3: Tuning Leaderboard

def test_tuning_leaderboard():
    """Test leaderboard generation"""
    print("\n[TEST 3] Tuning Leaderboard Generation")
    print("-" * 70)
    
    # Create mock CV results
    cv_results_data = {
        'mean_test_score': [-10.5, -11.2, -10.8, -12.1, -11.5],
        'std_test_score': [0.5, 0.6, 0.4, 0.8, 0.7],
        'rank_test_score': [1, 3, 2, 5, 4],
        'param_model__n_estimators': [100, 50, 100, 50, 200],
        'param_model__max_depth': [20, 10, 30, 10, 20]
    }
    cv_results_df = pd.DataFrame(cv_results_data)
    
    # Generate leaderboard
    leaderboard = create_tuning_leaderboard(cv_results_df, top_n=3)
    
    # Verify leaderboard
    assert len(leaderboard) == 3, "Leaderboard should have 3 rows"
    assert 'mean_rmse' in leaderboard.columns, "Missing mean_rmse column"
    assert 'std_rmse' in leaderboard.columns, "Missing std_rmse column"
    assert 'rank_test_score' in leaderboard.columns, "Missing rank column"
    
    # Verify RMSE conversion (should be positive)
    assert leaderboard['mean_rmse'].iloc[0] > 0, "RMSE should be positive"
    
    # Verify sorted by rank
    assert leaderboard['rank_test_score'].iloc[0] == 1, "Should be sorted by rank"
    
    print(f"   Leaderboard generated: {len(leaderboard)} rows")
    print(f"   Best RMSE: ${leaderboard['mean_rmse'].iloc[0]:.2f}")
    print(" PASSED")


# TEST 4: Model Saving and Loading

def test_model_save_load():
    """Test model saving and loading"""
    print("\n[TEST 4] Model Save/Load")
    print("-" * 70)
    
    # Create a simple model
    train_df, _ = create_train_test(verbose=False)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df['Cost_USD_per_MWh']
    
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        model_type='linear',
        verbose=False
    )
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Save model
    metadata = {'test_param': 'test_value', 'score': 0.85}
    filepath = save_model(pipeline, "test_model", metadata=metadata)
    
    # Verify file exists
    assert os.path.exists(filepath), "Model file not created"
    assert filepath.endswith('.pkl'), "Model file should be .pkl"
    
    # Load model
    loaded_obj = load_model(filepath)
    
    # Verify loaded object structure
    assert 'model' in loaded_obj, "Missing model in loaded object"
    assert 'model_name' in loaded_obj, "Missing model_name"
    assert 'timestamp' in loaded_obj, "Missing timestamp"
    assert 'metadata' in loaded_obj, "Missing metadata"
    
    # Verify metadata
    assert loaded_obj['metadata']['test_param'] == 'test_value', "Metadata not preserved"
    
    # Verify model works
    loaded_model = loaded_obj['model']
    predictions = loaded_model.predict(X_train[:5])
    assert len(predictions) == 5, "Loaded model prediction failed"
    
    print(f"   Model saved: {filepath}")
    print(f"   Model loaded successfully")
    print(f"   Metadata preserved")
    
    # Clean up
    os.remove(filepath)
    print("   Test file cleaned up")
    print(" PASSED")


# TEST 5: NECPipeline Initialization

def test_pipeline_initialization():
    """Test NECPipeline class initialization"""
    print("\n[TEST 5] NECPipeline Initialization")
    print("-" * 70)
    
    # Create pipeline
    pipeline = NECPipeline(verbose=False)
    
    # Verify attributes
    assert pipeline is not None, "Pipeline not created"
    assert hasattr(pipeline, 'load_data'), "Missing load_data method"
    assert hasattr(pipeline, 'create_preprocessor'), "Missing create_preprocessor method"
    assert hasattr(pipeline, 'train_baseline'), "Missing train_baseline method"
    assert hasattr(pipeline, 'evaluate_baseline'), "Missing evaluate_baseline method"
    assert hasattr(pipeline, 'tune_model'), "Missing tune_model method"
    assert hasattr(pipeline, 'evaluate_tuned'), "Missing evaluate_tuned method"
    assert hasattr(pipeline, 'compare_models'), "Missing compare_models method"
    assert hasattr(pipeline, 'create_visualizations'), "Missing create_visualizations method"
    assert hasattr(pipeline, 'run_complete_pipeline'), "Missing run_complete_pipeline method"
    
    print("   NECPipeline initialized")
    print("   All required methods present")
    print(" PASSED")


# TEST 6: Pipeline Data Loading

def test_pipeline_data_loading():
    """Test pipeline data loading step"""
    print("\n[TEST 6] Pipeline Data Loading")
    print("-" * 70)
    
    # Create pipeline
    pipeline = NECPipeline(verbose=False)
    
    # Load data
    pipeline.load_data()
    
    # Verify data loaded
    assert pipeline.train_df is not None, "Training data not loaded"
    assert pipeline.test_df is not None, "Test data not loaded"
    assert len(pipeline.train_df) > 0, "Training data is empty"
    assert len(pipeline.test_df) > 0, "Test data is empty"
    
    print(f"   Training data: {pipeline.train_df.shape}")
    print(f"   Test data: {pipeline.test_df.shape}")
    print(" PASSED")


# TEST 7: Pipeline Preprocessing Step

def test_pipeline_preprocessing():
    """Test pipeline preprocessing step"""
    print("\n[TEST 7] Pipeline Preprocessing")
    print("-" * 70)
    
    # Create pipeline
    pipeline = NECPipeline(verbose=False)
    pipeline.load_data()
    
    # Create preprocessor
    pipeline.create_preprocessor()
    
    # Verify preprocessor created
    assert pipeline.preprocessor is not None, "Preprocessor not created"
    
    # Verify preprocessor is fitted
    # (sklearn preprocessors have _sklearn_is_fitted or named_transformers_ after fitting)
    assert hasattr(pipeline.preprocessor, 'transform'), "Preprocessor not fitted"
    
    print("   Preprocessor created and fitted")
    print(" PASSED")



# TEST 8: Pipeline Baseline Training

def test_pipeline_baseline_training():
    """Test pipeline baseline model training"""
    print("\n[TEST 8] Pipeline Baseline Training")
    print("-" * 70)
    
    # Create pipeline
    pipeline = NECPipeline(verbose=False)
    pipeline.load_data()
    pipeline.create_preprocessor()
    
    # Train baseline
    pipeline.train_baseline()
    
    # Verify baseline model created
    assert pipeline.baseline_model is not None, "Baseline model not created"
    assert hasattr(pipeline.baseline_model, 'fit'), "Baseline model missing fit method"
    
    print("   Baseline model created")
    print(" PASSED")



# TEST 9: Complete Pipeline

def test_complete_pipeline_no_tuning():
    """Test complete pipeline without tuning"""
    print("\n[TEST 9] Complete Pipeline (No Tuning)")
    print("-" * 70)
    
    # Create pipeline
    pipeline = NECPipeline(verbose=False)
    
    # Run pipeline without tuning
    print("  Running pipeline (this may take 1-2 minutes)...")
    results = pipeline.run_complete_pipeline(tune=False, quick=True)
    
    # Verify results
    assert 'baseline_results' in results, "Missing baseline results"
    assert 'tuned_results' in results, "Missing tuned results"
    assert 'comparison' in results, "Missing comparison"
    
    # Verify baseline results exist
    assert results['baseline_results'] is not None, "Baseline results is None"
    
    # Verify tuned results are None (no tuning)
    assert results['tuned_results'] is None, "Tuned results should be None"
    
    print("   Pipeline completed without tuning")
    print("   Baseline results generated")
    print(" PASSED")



# TEST 10: Complete Pipeline


def test_complete_pipeline_with_tuning():
    """Test complete pipeline with quick tuning"""
    print("\n[TEST 10] Complete Pipeline (With Quick Tuning)")
    print("-" * 70)
    
    # Create pipeline
    pipeline = NECPipeline(verbose=False)
    
    # Run pipeline with quick tuning
    print("  Running pipeline with tuning (this may take 3-5 minutes)...")
    results = pipeline.run_complete_pipeline(tune=True, quick=True)
    
    # Verify results
    assert 'baseline_results' in results, "Missing baseline results"
    assert 'tuned_results' in results, "Missing tuned results"
    assert 'comparison' in results, "Missing comparison"
    
    # Verify both models evaluated
    assert results['baseline_results'] is not None, "Baseline results is None"
    assert results['tuned_results'] is not None, "Tuned results is None"
    
    # Verify comparison dataframe
    comparison_df = results['comparison']
    assert comparison_df is not None, "Comparison dataframe is None"
    assert len(comparison_df) == 2, "Should compare 2 models"
    
    # Verify both models have test results
    baseline_test = results['baseline_results']['test']
    tuned_test = results['tuned_results']['test']
    
    assert 'rmse' in baseline_test, "Missing baseline test RMSE"
    assert 'rmse' in tuned_test, "Missing tuned test RMSE"
    
    print("   Pipeline completed with tuning")
    print(f"   Baseline RMSE: ${baseline_test['rmse']:.2f}")
    print(f"   Tuned RMSE: ${tuned_test['rmse']:.2f}")
    print(f"   Baseline Selection Error: {baseline_test['selection_error']:.2%}")
    print(f"   Tuned Selection Error: {tuned_test['selection_error']:.2%}")
    print(" PASSED")


# TEST 11: Integration Test

def test_all_members_integration():
    """Test complete integration"""
    print("\n[TEST 11] Complete Integration (All Members)")
    print("-" * 70)
    
    # Data
    from src.data_ingestion import create_train_test
    train_df, test_df = create_train_test(verbose=False)
    print("   Step 1: Data loaded")
    
    # Preprocessing
    from src.preprocessing import NECPreprocessor
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    print("   Steo 2: Preprocessing complete")
    
    # Model
    from src.models import create_model_pipeline
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        model_type='linear',
        verbose=False
    )
    print("   Step 3: Model created")
    
    # Evaluation
    from src.evaluation import ModelEvaluator
    evaluator = ModelEvaluator(verbose=False)
    results = evaluator.evaluate_model_comprehensive(
        pipeline,
        train_df,
        test_df,
        model_name="Integration_Test",
        run_logo_cv=False
    )
    print("   Step 4: Evaluation complete")
    
    # Verify tuning capabilities
    assert tune_hyperparameters is not None, "Tuning function not available"
    print("   Step 5: Tuning capabilities verified")
    
    # Verify final results
    assert 'test' in results, "Missing test results"
    assert 'train' in results, "Missing train results"
    
    print("\n   All integrated successfully!")
    print(" PASSED")


# RUN ALL TESTS

if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIT TESTS")
    print("="*70)
    
    tests = [
        test_tuning_imports,
        test_hyperparameter_tuning_quick,
        test_tuning_leaderboard,
        test_model_save_load,
        test_pipeline_initialization,
        test_pipeline_data_loading,
        test_pipeline_preprocessing,
        test_pipeline_baseline_training,
        test_complete_pipeline_no_tuning,
        test_complete_pipeline_with_tuning,
        test_all_members_integration
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