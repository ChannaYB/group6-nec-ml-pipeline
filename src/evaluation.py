"""
Evaluation Framework for NEC ML Pipeline
Comprehensive model evaluation with LOGO CV and reporting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import clone

from src.models import evaluate_with_logo_cv, evaluate_model
from src.custom_scorer import get_selection_error_table
from src.config import (
    LOGO_CV_SPLITS,
    EVALUATION_REPORTS_DIR,
    SELECTION_TABLES_DIR,
    GROUP_COLUMN,
    TARGET_COLUMN,
    VERBOSE
)


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    Provides:
    - LOGO Cross-Validation
    - Train/Test evaluation
    - Fold-level analysis
    - Selection error tables
    - Comparison reports
    
    Reference: Assessment Brief - "Evaluation outcomes (grouped train/test and LOGO)"
    """
    def __init__(self, verbose=VERBOSE):
        """Initialize evaluator."""
        self.verbose = verbose
        self.results = {}
        
    def evaluate_model_comprehensive(
        self,
        pipeline,
        train_df,
        test_df,
        model_name="Model",
        run_logo_cv=True
    ):
        """
        Run comprehensive evaluation on a model.
        
        Steps:
        1. LOGO Cross-Validation on training data
        2. Fit pipeline on full training data
        3. Train set evaluation
        4. Test set evaluation
        5. Generate selection error tables
        
        Parameters:
        -----------
        pipeline : sklearn Pipeline
            Unfitted pipeline (will be cloned for CV, then fitted for train/test)
        train_df : pandas.DataFrame
            Training data
        test_df : pandas.DataFrame
            Testing data
        model_name : str
            Name for reporting
        run_logo_cv : bool
            Whether to run LOGO CV (slow for large models)
        
        Returns:
        --------
        dict : Comprehensive evaluation results
        
        Reference: Assessment Brief - "Grouped train/test and LOGO CV"
        """

        if self.verbose:
            print("\n" + "="*70)
            print(f"COMPREHENSIVE EVALUATION: {model_name}")
            print("="*70)
        
        from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
        
        feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Prepare data
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COLUMN]
        
        # 1. LOGO Cross-Validation (uses unfitted pipeline clones)
        if run_logo_cv:
            if self.verbose:
                print(f"\n[1] Running LOGO Cross-Validation...")
            
            cv_results = evaluate_with_logo_cv(
                pipeline,  # Pass unfitted pipeline
                train_df,
                n_splits=LOGO_CV_SPLITS,
                verbose=self.verbose
            )
            results['logo_cv'] = cv_results
        else:
            if self.verbose:
                print(f"\n[1] Skipping LOGO CV (run_logo_cv=False)")
            results['logo_cv'] = None
        
        # 2. Fit pipeline on FULL training data for train/test evaluation
        if self.verbose:
            print(f"\n[2] Fitting on Full Training Set...")
        
        from src.models import train_model
        
        # Clone pipeline to avoid issues
        fitted_pipeline = clone(pipeline)
        fitted_pipeline = train_model(fitted_pipeline, X_train, y_train, verbose=self.verbose)
        
        # 3. Train set evaluation
        if self.verbose:
            print(f"\n[3] Evaluating on Training Set...")
        
        train_results = evaluate_model(
            fitted_pipeline,
            X_train,
            y_train,
            train_df[GROUP_COLUMN],
            train_df['Plant ID'],
            dataset_name="Train",
            verbose=self.verbose
        )
        results['train'] = train_results
        
        # 4. Test set evaluation
        if self.verbose:
            print(f"\n[4] Evaluating on Test Set...")
        
        test_results = evaluate_model(
            fitted_pipeline,
            X_test,
            y_test,
            test_df[GROUP_COLUMN],
            test_df['Plant ID'],
            dataset_name="Test",
            verbose=self.verbose
        )
        results['test'] = test_results
        
        # 5. Generate selection error tables
        if self.verbose:
            print(f"\n[5] Generating Selection Error Tables...")
        
        # Test set selection table
        y_test_pred = fitted_pipeline.predict(X_test)
        test_selection_table = get_selection_error_table(
            y_test,
            y_test_pred,
            test_df[GROUP_COLUMN],
            test_df['Plant ID']
        )
        results['test_selection_table'] = test_selection_table
        
        # Train set selection table (optional, for analysis)
        y_train_pred = fitted_pipeline.predict(X_train)
        train_selection_table = get_selection_error_table(
            y_train,
            y_train_pred,
            train_df[GROUP_COLUMN],
            train_df['Plant ID']
        )
        results['train_selection_table'] = train_selection_table
        
        if self.verbose:
            print(f"   Test selection table: {len(test_selection_table)} demands")
            print(f"   Train selection table: {len(train_selection_table)} demands")
        
        # Store fitted pipeline for later use
        results['fitted_pipeline'] = fitted_pipeline
        
        # Store in instance
        self.results[model_name] = results
        
        if self.verbose:
            print("\n" + "="*70)
            print(f" EVALUATION COMPLETE: {model_name}")
            print("="*70)
        
        return results
    
    def compare_models(self, model_results_dict):
        """
        Compare multiple models.
        
        Parameters:
        -----------
        model_results_dict : dict
            Dictionary of {model_name: results}
        
        Returns:
        --------
        pandas.DataFrame : Comparison table
        
        Reference: Assessment Brief - "Comparison with untuned baseline"
        """

        if self.verbose:
            print("\n" + "="*70)
            print("MODEL COMPARISON")
            print("="*70)
        
        comparison_data = []
        
        for model_name, results in model_results_dict.items():
            row = {'Model': model_name}
            
            # LOGO CV metrics (if available)
            if results.get('logo_cv'):
                cv = results['logo_cv']
                row['CV_RMSE'] = cv['avg_rmse']
                row['CV_RMSE_Std'] = cv['std_rmse']
                row['CV_Selection_Error'] = cv['avg_selection_error']
                row['CV_Selection_Error_Std'] = cv['std_selection_error']
            
            # Train metrics
            if results.get('train'):
                row['Train_RMSE'] = results['train']['rmse']
                row['Train_R2'] = results['train']['r2']
                row['Train_Selection_Error'] = results['train']['selection_error']
            
            # Test metrics
            if results.get('test'):
                row['Test_RMSE'] = results['test']['rmse']
                row['Test_R2'] = results['test']['r2']
                row['Test_Selection_Error'] = results['test']['selection_error']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if self.verbose:
            print("\n", comparison_df.to_string(index=False))
            print("\n" + "="*70)
        
        return comparison_df
    
    def generate_fold_analysis(self, cv_results, model_name="Model"):
        """
        Analyze performance across CV folds.
        
        Parameters:
        -----------
        cv_results : dict
            Results from LOGO CV
        model_name : str
            Model name for reporting
        
        Returns:
        --------
        pandas.DataFrame : Fold-level metrics
        
        Reference: Assessment Brief - "Fold-level outputs"
        """
        if cv_results is None:
            return None
        
        fold_data = []
        
        for i, fold_result in enumerate(cv_results['fold_results']):
            fold_data.append({
                'Fold': i + 1,
                'RMSE': fold_result['rmse'],
                'R²': fold_result['r2'],
                'Selection_Error': fold_result['selection_error'],
                'N_Demands': fold_result['n_demands'],
                'N_Samples': fold_result['n_samples']
            })
        
        fold_df = pd.DataFrame(fold_data)
        
        # Add summary statistics
        summary = {
            'Fold': 'AVERAGE',
            'RMSE': fold_df['RMSE'].mean(),
            'R²': fold_df['R²'].mean(),
            'Selection_Error': fold_df['Selection_Error'].mean(),
            'N_Demands': fold_df['N_Demands'].mean(),
            'N_Samples': fold_df['N_Samples'].mean()
        }
        fold_df = pd.concat([fold_df, pd.DataFrame([summary])], ignore_index=True)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"FOLD-LEVEL ANALYSIS: {model_name}")
            print(f"{'='*70}")
            print("\n", fold_df.to_string(index=False))
            print(f"\n{'='*70}")
        
        return fold_df
    
    def save_evaluation_report(self, results, model_name="Model"):
        """
        Save comprehensive evaluation report to file.
        
        Parameters:
        -----------
        results : dict
            Evaluation results
        model_name : str
            Model name
        
        Returns:
        --------
        str : Path to saved report
        
        Reference: Assessment Brief - "Artefacts and reproducibility"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{model_name.replace(' ', '_')}_{timestamp}.txt"
        filepath = Path(EVALUATION_REPORTS_DIR) / filename
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"NEC ML PIPELINE - EVALUATION REPORT\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # LOGO CV Results
            if results.get('logo_cv'):
                cv = results['logo_cv']
                f.write(f"LOGO CROSS-VALIDATION ({cv['n_splits']} folds)\n")
                f.write("-"*70 + "\n")
                f.write(f"Average RMSE: ${cv['avg_rmse']:.2f} ± ${cv['std_rmse']:.2f}\n")
                f.write(f"Average R²: {cv['avg_r2']:.4f}\n")
                f.write(f"Average Selection Error: {cv['avg_selection_error']:.2%} ± {cv['std_selection_error']:.2%}\n\n")
                
                # Fold details
                f.write("Fold-Level Results:\n")
                for i, fold in enumerate(cv['fold_results']):
                    f.write(f"  Fold {i+1}: RMSE=${fold['rmse']:.2f}, R²={fold['r2']:.4f}, Error={fold['selection_error']:.2%}\n")
                f.write("\n")
            
            # Train Results
            if results.get('train'):
                train = results['train']
                f.write("TRAINING SET EVALUATION\n")
                f.write("-"*70 + "\n")
                f.write(f"RMSE: ${train['rmse']:.2f}/MWh\n")
                f.write(f"R²: {train['r2']:.4f}\n")
                f.write(f"Selection Error: {train['selection_error']:.2%}\n")
                f.write(f"Samples: {train['n_samples']:,}\n")
                f.write(f"Demands: {train['n_demands']}\n\n")
            
            # Test Results
            if results.get('test'):
                test = results['test']
                f.write("TEST SET EVALUATION\n")
                f.write("-"*70 + "\n")
                f.write(f"RMSE: ${test['rmse']:.2f}/MWh\n")
                f.write(f"R²: {test['r2']:.4f}\n")
                f.write(f"Selection Error: {test['selection_error']:.2%}\n")
                f.write(f"Samples: {test['n_samples']:,}\n")
                f.write(f"Demands: {test['n_demands']}\n\n")
            
            f.write("="*70 + "\n")
        
        if self.verbose:
            print(f"\n Evaluation report saved: {filepath}")
        
        return str(filepath)
    
    def save_selection_table(self, selection_table, model_name="Model", dataset_name="Test"):
        """
        Save selection error table to CSV.
        
        Parameters:
        -----------
        selection_table : pandas.DataFrame
            Selection error table
        model_name : str
            Model name
        dataset_name : str
            Dataset name (Train/Test)
        
        Returns:
        --------
        str : Path to saved file
        
        Reference: Assessment Brief - "Per-scenario selection table"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selection_{model_name.replace(' ', '_')}_{dataset_name}_{timestamp}.csv"
        filepath = Path(SELECTION_TABLES_DIR) / filename
        
        selection_table.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f" Selection table saved: {filepath}")
        
        return str(filepath)


# CONVENIENCE FUNCTIONS

def evaluate_baseline_model(train_df, test_df, verbose=VERBOSE):
    """
    Evaluate baseline (untuned) Random Forest model.
    
    This provides the comparison benchmark for tuned models.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data
    test_df : pandas.DataFrame
        Test data
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict : Evaluation results
    
    Reference: Assessment Brief - "Comparison with the untuned baseline"
    """
    from src.preprocessing import NECPreprocessor
    from src.models import create_model_pipeline
    from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    
    if verbose:
        print("\n" + "="*70)
        print("EVALUATING BASELINE MODEL (Untuned Random Forest)")
        print("="*70)
    
    # Prepare data
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    
    # Create preprocessor
    preprocessor_obj = NECPreprocessor(verbose=False)
    preprocessor_obj.fit(X_train)
    
    # Create baseline pipeline
    pipeline = create_model_pipeline(
        preprocessor_obj.preprocessor,
        model_type='random_forest',
        verbose=verbose
    )
    
    # Evaluate
    evaluator = ModelEvaluator(verbose=verbose)
    results = evaluator.evaluate_model_comprehensive(
        pipeline,
        train_df,
        test_df,
        model_name="Baseline_RF",
        run_logo_cv=True
    )
    
    # Save reports
    evaluator.save_evaluation_report(results, "Baseline_RF")
    evaluator.save_selection_table(
        results['test_selection_table'],
        "Baseline_RF",
        "Test"
    )
    
    return results


# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING EVALUATION MODULE")
    print("="*70)
    
    try:
        # Load data
        from src.data_ingestion import create_train_test
        
        print("\n[1] Loading data...")
        train_df, test_df = create_train_test(verbose=False)
        print(f" Data loaded: Train {train_df.shape}, Test {test_df.shape}")
        
        # Evaluate baseline model
        print("\n[2] Evaluating baseline model...")
        results = evaluate_baseline_model(train_df, test_df, verbose=True)
        
        # Generate fold analysis
        print("\n[3] Generating fold analysis...")
        evaluator = ModelEvaluator(verbose=True)
        fold_df = evaluator.generate_fold_analysis(
            results['logo_cv'],
            "Baseline_RF"
        )
        
        print("\n" + "="*70)
        print(" EVALUATION MODULE TEST PASSED")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n Error: {e}\n")
        import traceback
        traceback.print_exc()