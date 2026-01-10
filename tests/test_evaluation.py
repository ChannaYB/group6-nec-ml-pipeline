"""
Unit tests: Evaluation and Visualization
Tests evaluation framework, reporting, and visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os

from src.data_ingestion import create_train_test
from src.preprocessing import NECPreprocessor
from src.models import create_model_pipeline
from src.evaluation import ModelEvaluator, evaluate_baseline_model
from src.visualization import (
    plot_logo_cv_folds,
    plot_model_comparison,
    plot_prediction_scatter,
    plot_selection_error_distribution,
    create_evaluation_dashboard
)
from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    EVALUATION_REPORTS_DIR,
    SELECTION_TABLES_DIR,
    PLOTS_DIR
)


# TEST 1: Model Evaluator Initialization

def test_evaluator_initialization():
    """Test that ModelEvaluator initializes correctly"""
    print("\n[TEST 1] ModelEvaluator Initialization")
    print("-" * 70)
    
    evaluator = ModelEvaluator(verbose=False)
    
    assert evaluator is not None, "Evaluator not created"
    assert hasattr(evaluator, 'results'), "Missing results attribute"
    assert hasattr(evaluator, 'evaluate_model_comprehensive'), "Missing evaluation method"
    
    print("   Evaluator initialized")
    print("   All required attributes present")
    print(" PASSED")


# TEST 2: Comprehensive Evaluation

def test_comprehensive_evaluation():
    """Test comprehensive model evaluation"""
    print("\n[TEST 2] Comprehensive Evaluation")
    print("-" * 70)
    
    # Load small dataset
    train_df, test_df = create_train_test(verbose=False)
    
    # Create simple pipeline
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        model_type='linear',  # Use Linear for speed
        verbose=False
    )
    
    # Evaluate (without LOGO CV for speed)
    evaluator = ModelEvaluator(verbose=False)
    results = evaluator.evaluate_model_comprehensive(
        pipeline,
        train_df,
        test_df,
        model_name="Test_Model",
        run_logo_cv=False  # Skip CV for faster testing
    )
    
    # Verify results structure
    assert 'model_name' in results, "Missing model_name"
    assert 'train' in results, "Missing train results"
    assert 'test' in results, "Missing test results"
    assert 'test_selection_table' in results, "Missing selection table"
    
    # Verify train results
    assert 'rmse' in results['train'], "Missing train RMSE"
    assert 'selection_error' in results['train'], "Missing train selection error"
    
    # Verify test results
    assert 'rmse' in results['test'], "Missing test RMSE"
    assert 'selection_error' in results['test'], "Missing test selection error"
    
    # Verify selection table
    assert len(results['test_selection_table']) > 0, "Empty selection table"
    
    print(f"   Evaluation completed")
    print(f"   Train RMSE: ${results['train']['rmse']:.2f}")
    print(f"   Test RMSE: ${results['test']['rmse']:.2f}")
    print(f"   Selection table: {len(results['test_selection_table'])} demands")
    print(" PASSED")


# TEST 3: LOGO CV Integration


    """Test LOGO CV evaluation"""
    print("\n[TEST 3] LOGO CV Evaluation")
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
        model_type='linear',  # Use Linear for speed
        verbose=False
    )
    
    # Evaluate with LOGO CV (use 3 folds directly in function call)
    evaluator = ModelEvaluator(verbose=False)
    
    # Manually run LOGO CV with 3 folds
    from src.models import evaluate_with_logo_cv
    cv_results = evaluate_with_logo_cv(
        pipeline,
        train_df,
        n_splits=3,  # Directly specify 3 folds
        verbose=False
    )
    
    # Create results dict
    results = {
        'model_name': 'Test_CV_Model',
        'logo_cv': cv_results,
        'train': None,
        'test': None
    }
    
    # Verify CV results
    assert results['logo_cv'] is not None, "LOGO CV results missing"
    assert 'n_splits' in results['logo_cv'], "Missing n_splits"
    assert 'fold_results' in results['logo_cv'], "Missing fold_results"
    assert 'avg_rmse' in results['logo_cv'], "Missing avg_rmse"
    assert 'avg_selection_error' in results['logo_cv'], "Missing avg_selection_error"
    
    # Verify fold count
    assert results['logo_cv']['n_splits'] == 3, f"Wrong number of folds: expected 3, got {results['logo_cv']['n_splits']}"
    assert len(results['logo_cv']['fold_results']) == 3, f"Wrong number of fold results: expected 3, got {len(results['logo_cv']['fold_results'])}"
    
    print(f"   LOGO CV completed with {results['logo_cv']['n_splits']} folds")
    print(f"   Average RMSE: ${results['logo_cv']['avg_rmse']:.2f}")
    print(f"   Average Selection Error: {results['logo_cv']['avg_selection_error']:.2%}")
    print(" PASSED")
    """Test LOGO CV evaluation"""
    print("\n[TEST 3] LOGO CV Evaluation")
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
        model_type='linear',  # Use Linear for speed
        verbose=False
    )
    
    # Evaluate with LOGO CV (3 folds for speed)
    from src.config import LOGO_CV_SPLITS
    import src.config as config
    original_splits = config.LOGO_CV_SPLITS
    config.LOGO_CV_SPLITS = 3  # Temporarily reduce for testing
    
    evaluator = ModelEvaluator(verbose=False)
    results = evaluator.evaluate_model_comprehensive(
        pipeline,
        train_df,
        test_df,
        model_name="Test_CV_Model",
        run_logo_cv=True
    )
    
    # Restore original
    config.LOGO_CV_SPLITS = original_splits
    
    # Verify CV results
    assert results['logo_cv'] is not None, "LOGO CV results missing"
    assert 'n_splits' in results['logo_cv'], "Missing n_splits"
    assert 'fold_results' in results['logo_cv'], "Missing fold_results"
    assert 'avg_rmse' in results['logo_cv'], "Missing avg_rmse"
    assert 'avg_selection_error' in results['logo_cv'], "Missing avg_selection_error"
    
    # Verify fold count
    assert results['logo_cv']['n_splits'] == 3, "Wrong number of folds"
    assert len(results['logo_cv']['fold_results']) == 3, "Wrong number of fold results"
    
    print(f"   LOGO CV completed with {results['logo_cv']['n_splits']} folds")
    print(f"   Average RMSE: ${results['logo_cv']['avg_rmse']:.2f}")
    print(f"   Average Selection Error: {results['logo_cv']['avg_selection_error']:.2%}")
    print(" PASSED")


    """Test LOGO CV evaluation with direct n_splits parameter"""
    print("\n[TEST 3] LOGO CV Evaluation")
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
        model_type='linear',  # Use Linear for speed
        verbose=False
    )
    
    # Run LOGO CV directly with 3 folds (for faster testing)
    from src.models import evaluate_with_logo_cv
    
    print("  Running LOGO CV with 3 folds...")
    cv_results = evaluate_with_logo_cv(
        pipeline,
        train_df,
        n_splits=3,
        verbose=False
    )
    
    # Verify CV results structure
    assert cv_results is not None, "LOGO CV results missing"
    assert 'n_splits' in cv_results, "Missing n_splits in results"
    assert 'fold_results' in cv_results, "Missing fold_results in results"
    assert 'avg_rmse' in cv_results, "Missing avg_rmse in results"
    assert 'avg_selection_error' in cv_results, "Missing avg_selection_error in results"
    
    # Verify fold count matches what we requested
    actual_folds = cv_results['n_splits']
    assert actual_folds == 3, f"Expected 3 folds, got {actual_folds}"
    
    # Verify we have results for each fold
    actual_fold_results = len(cv_results['fold_results'])
    assert actual_fold_results == 3, f"Expected 3 fold results, got {actual_fold_results}"
    
    # Verify each fold has required metrics
    for i, fold in enumerate(cv_results['fold_results']):
        assert 'rmse' in fold, f"Fold {i+1} missing RMSE"
        assert 'selection_error' in fold, f"Fold {i+1} missing selection_error"
        assert 'r2' in fold, f"Fold {i+1} missing R²"
    
    print(f"   LOGO CV completed with {cv_results['n_splits']} folds")
    print(f"   Average RMSE: ${cv_results['avg_rmse']:.2f}")
    print(f"   Average Selection Error: {cv_results['avg_selection_error']:.2%}")
    print(f"   All {len(cv_results['fold_results'])} folds have complete metrics")
    print(" PASSED")

# TEST 3: LOGO CV Integration

def test_logo_cv_evaluation():
    """Test LOGO CV evaluation with direct n_splits parameter"""
    print("\n[TEST 3] LOGO CV Evaluation")
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
        model_type='linear',  # Use Linear for speed
        verbose=False
    )
    
    # Run LOGO CV directly with 3 folds (for faster testing)
    from src.models import evaluate_with_logo_cv
    
    print("  Running LOGO CV with 3 folds...")
    cv_results = evaluate_with_logo_cv(
        pipeline,
        train_df,
        n_splits=3,
        verbose=False
    )
    
    # Verify CV results structure
    assert cv_results is not None, "LOGO CV results missing"
    assert 'n_splits' in cv_results, "Missing n_splits in results"
    assert 'fold_results' in cv_results, "Missing fold_results in results"
    assert 'avg_rmse' in cv_results, "Missing avg_rmse in results"
    assert 'avg_selection_error' in cv_results, "Missing avg_selection_error in results"
    
    # Verify fold count matches what we requested
    actual_folds = cv_results['n_splits']
    assert actual_folds == 3, f"Expected 3 folds, got {actual_folds}"
    
    # Verify we have results for each fold
    actual_fold_results = len(cv_results['fold_results'])
    assert actual_fold_results == 3, f"Expected 3 fold results, got {actual_fold_results}"
    
    # Verify each fold has required metrics
    for i, fold in enumerate(cv_results['fold_results']):
        assert 'rmse' in fold, f"Fold {i+1} missing RMSE"
        assert 'selection_error' in fold, f"Fold {i+1} missing selection_error"
        assert 'r2' in fold, f"Fold {i+1} missing R²"
    
    print(f"   LOGO CV completed with {cv_results['n_splits']} folds")
    print(f"   Average RMSE: ${cv_results['avg_rmse']:.2f}")
    print(f"   Average Selection Error: {cv_results['avg_selection_error']:.2%}")
    print(f"   All {len(cv_results['fold_results'])} folds have complete metrics")
    print(" PASSED")


# TEST 4: Model Comparison

def test_model_comparison():
    """Test model comparison functionality"""
    print("\n[TEST 4] Model Comparison")
    print("-" * 70)
    
    # Create mock results for two models
    results_dict = {
        'Model_A': {
            'train': {'rmse': 10.0, 'r2': 0.8, 'selection_error': 0.3},
            'test': {'rmse': 12.0, 'r2': 0.7, 'selection_error': 0.4},
            'logo_cv': {
                'avg_rmse': 11.0,
                'std_rmse': 1.0,
                'avg_selection_error': 0.35,
                'std_selection_error': 0.05
            }
        },
        'Model_B': {
            'train': {'rmse': 9.0, 'r2': 0.85, 'selection_error': 0.25},
            'test': {'rmse': 11.0, 'r2': 0.75, 'selection_error': 0.35},
            'logo_cv': {
                'avg_rmse': 10.0,
                'std_rmse': 0.8,
                'avg_selection_error': 0.30,
                'std_selection_error': 0.04
            }
        }
    }
    
    # Compare models
    evaluator = ModelEvaluator(verbose=False)
    comparison_df = evaluator.compare_models(results_dict)
    
    # Verify comparison table
    assert len(comparison_df) == 2, "Should have 2 models"
    assert 'Model' in comparison_df.columns, "Missing Model column"
    assert 'Test_RMSE' in comparison_df.columns, "Missing Test_RMSE column"
    assert 'CV_Selection_Error' in comparison_df.columns, "Missing CV_Selection_Error column"
    
    print(f"   Comparison table created: {len(comparison_df)} models")
    print(f"   Columns: {list(comparison_df.columns)}")
    print(" PASSED")


# TEST 5: Fold Analysis

def test_fold_analysis():
    """Test fold-level analysis"""
    print("\n[TEST 5] Fold Analysis")
    print("-" * 70)
    
    # Create mock CV results
    cv_results = {
        'n_splits': 3,
        'fold_results': [
            {'rmse': 10.0, 'r2': 0.8, 'selection_error': 0.3, 'n_demands': 50, 'n_samples': 3200},
            {'rmse': 11.0, 'r2': 0.75, 'selection_error': 0.35, 'n_demands': 50, 'n_samples': 3200},
            {'rmse': 9.5, 'r2': 0.82, 'selection_error': 0.28, 'n_demands': 50, 'n_samples': 3200}
        ],
        'avg_rmse': 10.17,
        'avg_selection_error': 0.31
    }
    
    # Generate fold analysis
    evaluator = ModelEvaluator(verbose=False)
    fold_df = evaluator.generate_fold_analysis(cv_results, "Test_Model")
    
    # Verify fold analysis
    assert len(fold_df) == 4, "Should have 3 folds + 1 average row"
    assert 'Fold' in fold_df.columns, "Missing Fold column"
    assert 'RMSE' in fold_df.columns, "Missing RMSE column"
    assert 'Selection_Error' in fold_df.columns, "Missing Selection_Error column"
    
    # Check average row
    assert fold_df.iloc[-1]['Fold'] == 'AVERAGE', "Missing average row"
    
    print(f"   Fold analysis created: {len(fold_df)} rows")
    print(f"   Includes average statistics")
    print(" PASSED")


# TEST 6: Report Saving

def test_report_saving():
    """Test evaluation report saving"""
    print("\n[TEST 6] Report Saving")
    print("-" * 70)
    
    # Create mock results
    results = {
        'model_name': 'Test_Model',
        'train': {'rmse': 10.0, 'r2': 0.8, 'selection_error': 0.3, 'n_samples': 10000, 'n_demands': 100},
        'test': {'rmse': 12.0, 'r2': 0.7, 'selection_error': 0.4, 'n_samples': 2500, 'n_demands': 25},
        'logo_cv': {
            'n_splits': 3,
            'avg_rmse': 11.0,
            'std_rmse': 1.0,
            'avg_r2': 0.75,
            'avg_selection_error': 0.35,
            'std_selection_error': 0.05,
            'fold_results': [
                {'rmse': 10.0, 'r2': 0.8, 'selection_error': 0.3},
                {'rmse': 11.0, 'r2': 0.75, 'selection_error': 0.35},
                {'rmse': 12.0, 'r2': 0.7, 'selection_error': 0.4}
            ]
        }
    }
    
    # Save report
    evaluator = ModelEvaluator(verbose=False)
    filepath = evaluator.save_evaluation_report(results, "Test_Model")
    
    # Verify file exists
    assert os.path.exists(filepath), "Report file not created"
    assert filepath.endswith('.txt'), "Report should be .txt file"
    
    # Read and verify content
    with open(filepath, 'r') as f:
        content = f.read()
        assert 'Test_Model' in content, "Missing model name in report"
        assert 'LOGO CROSS-VALIDATION' in content, "Missing CV section"
        assert 'TRAINING SET' in content, "Missing train section"
        assert 'TEST SET' in content, "Missing test section"
    
    print(f"   Report saved: {filepath}")
    print(f"   Report contains all required sections")
    
    # Clean up
    os.remove(filepath)
    print("   Test file cleaned up")
    print(" PASSED")


# TEST 7: Selection Table Saving

def test_selection_table_saving():
    """Test selection table saving"""
    print("\n[TEST 7] Selection Table Saving")
    print("-" * 70)
    
    # Create mock selection table
    selection_table = pd.DataFrame({
        'Demand ID': ['D1', 'D2', 'D3'],
        'True Best Plant': ['P1', 'P2', 'P3'],
        'Predicted Best Plant': ['P1', 'P3', 'P3'],
        'Selection Correct': [True, False, True],
        'True Best Cost': [10.0, 20.0, 30.0],
        'Predicted Cost': [10.5, 25.0, 30.5],
        'Cost Difference': [0.5, 5.0, 0.5]
    })
    
    # Save table
    evaluator = ModelEvaluator(verbose=False)
    filepath = evaluator.save_selection_table(selection_table, "Test_Model", "Test")
    
    # Verify file exists
    assert os.path.exists(filepath), "Selection table file not created"
    assert filepath.endswith('.csv'), "Table should be .csv file"
    
    # Read and verify content
    loaded_table = pd.read_csv(filepath)
    assert len(loaded_table) == 3, "Wrong number of rows"
    assert 'Demand ID' in loaded_table.columns, "Missing Demand ID column"
    assert 'Selection Correct' in loaded_table.columns, "Missing Selection Correct column"
    
    print(f"   Selection table saved: {filepath}")
    print(f"   Table has {len(loaded_table)} rows")
    
    # Clean up
    os.remove(filepath)
    print("   Test file cleaned up")
    print(" PASSED")


# TEST 8: Visualization - LOGO CV Plot

def test_visualization_logo_cv():
    """Test LOGO CV visualization"""
    print("\n[TEST 8] LOGO CV Visualization")
    print("-" * 70)
    
    # Create mock CV results
    cv_results = {
        'n_splits': 3,
        'fold_results': [
            {'rmse': 10.0, 'r2': 0.8, 'selection_error': 0.3, 'n_demands': 50, 'n_samples': 3200},
            {'rmse': 11.0, 'r2': 0.75, 'selection_error': 0.35, 'n_demands': 50, 'n_samples': 3200},
            {'rmse': 9.5, 'r2': 0.82, 'selection_error': 0.28, 'n_demands': 50, 'n_samples': 3200}
        ],
        'avg_rmse': 10.17,
        'std_rmse': 0.64,
        'avg_r2': 0.79,
        'avg_selection_error': 0.31,
        'std_selection_error': 0.03
    }
    
    # Create plot
    filepath = plot_logo_cv_folds(cv_results, "Test_Model", save=True, show=False)
    
    # Verify file exists
    assert filepath is not None, "Plot filepath not returned"
    assert os.path.exists(filepath), "Plot file not created"
    assert filepath.endswith('.png'), "Plot should be .png file"
    
    print(f"   LOGO CV plot created: {filepath}")
    
    # Clean up
    os.remove(filepath)
    print("   Test file cleaned up")
    print(" PASSED")


# TEST 9: Visualization - Model Comparison

def test_visualization_comparison():
    """Test model comparison visualization"""
    print("\n[TEST 9] Model Comparison Visualization")
    print("-" * 70)
    
    # Create mock comparison data
    comparison_df = pd.DataFrame({
        'Model': ['Model_A', 'Model_B'],
        'Test_RMSE': [12.0, 11.0],
        'Test_Selection_Error': [0.40, 0.35],
        'CV_Selection_Error': [0.35, 0.30],
        'CV_Selection_Error_Std': [0.05, 0.04]
    })
    
    # Create plot
    filepath = plot_model_comparison(comparison_df, save=True, show=False)
    
    # Verify file exists
    assert filepath is not None, "Plot filepath not returned"
    assert os.path.exists(filepath), "Plot file not created"
    assert filepath.endswith('.png'), "Plot should be .png file"
    
    print(f"   Comparison plot created: {filepath}")
    
    # Clean up
    os.remove(filepath)
    print("   Test file cleaned up")
    print(" PASSED")


# TEST 10: Visualization - Selection Error Distribution

def test_visualization_selection_error():
    """Test selection error distribution visualization"""
    print("\n[TEST 10] Selection Error Distribution Visualization")
    print("-" * 70)
    
    # Create mock selection table
    np.random.seed(42)
    n_demands = 50
    selection_table = pd.DataFrame({
        'Demand ID': [f'D{i}' for i in range(n_demands)],
        'True Best Plant': [f'P{i%10}' for i in range(n_demands)],
        'Predicted Best Plant': [f'P{(i+1)%10}' for i in range(n_demands)],
        'Selection Correct': np.random.choice([True, False], n_demands, p=[0.6, 0.4]),
        'True Best Cost': np.random.uniform(20, 80, n_demands),
        'Predicted Cost': np.random.uniform(20, 80, n_demands),
        'Cost Difference': np.random.uniform(-5, 10, n_demands)
    })
    
    # Create plot
    filepath = plot_selection_error_distribution(selection_table, "Test_Model", save=True, show=False)
    
    # Verify file exists
    assert filepath is not None, "Plot filepath not returned"
    assert os.path.exists(filepath), "Plot file not created"
    assert filepath.endswith('.png'), "Plot should be .png file"
    
    print(f"   Selection error plot created: {filepath}")
    
    # Clean up
    os.remove(filepath)
    print("   Test file cleaned up")
    print(" PASSED")


# TEST 11: Integration Test

def test_full_evaluation_pipeline():
    """Test complete evaluation pipeline"""
    print("\n[TEST 11] Full Evaluation Pipeline Integration")
    print("-" * 70)
    
    # Load data
    train_df, test_df = create_train_test(verbose=False)
    print("   Data loaded")
    
    # Create and evaluate baseline (without LOGO CV for speed)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        model_type='linear',
        verbose=False
    )
    
    evaluator = ModelEvaluator(verbose=False)
    results = evaluator.evaluate_model_comprehensive(
        pipeline,
        train_df,
        test_df,
        model_name="Integration_Test",
        run_logo_cv=False
    )
    print("   Model evaluated")
    
    # Save report
    report_path = evaluator.save_evaluation_report(results, "Integration_Test")
    assert os.path.exists(report_path), "Report not saved"
    print("   Report saved")
    
    # Save selection table
    table_path = evaluator.save_selection_table(
        results['test_selection_table'],
        "Integration_Test",
        "Test"
    )
    assert os.path.exists(table_path), "Selection table not saved"
    print("   Selection table saved")
    
    # Create visualization
    plot_path = plot_selection_error_distribution(
        results['test_selection_table'],
        "Integration_Test",
        save=True,
        show=False
    )
    assert os.path.exists(plot_path), "Plot not saved"
    print("   Visualization created")
    
    # Clean up
    os.remove(report_path)
    os.remove(table_path)
    os.remove(plot_path)
    print("   Test files cleaned up")
    
    print(" PASSED")


# RUN ALL TESTS

if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIT TESTS")
    print("="*70)
    
    tests = [
        test_evaluator_initialization,
        test_comprehensive_evaluation,
        test_logo_cv_evaluation,
        test_model_comparison,
        test_fold_analysis,
        test_report_saving,
        test_selection_table_saving,
        test_visualization_logo_cv,
        test_visualization_comparison,
        test_visualization_selection_error,
        test_full_evaluation_pipeline
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
        print("SOME TESTS FAILED")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print(" ALL TESTS PASSED!")
        print("="*70 + "\n")