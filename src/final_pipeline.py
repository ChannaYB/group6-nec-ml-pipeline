"""
Final Pipeline Integration for NEC ML Pipeline
Complete end-to-end system with all components
Reference: Assessment Brief - Final integration and delivery
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from src.data_ingestion import create_train_test
from src.preprocessing import NECPreprocessor
from src.models import create_model_pipeline, evaluate_with_logo_cv
from src.evaluation import ModelEvaluator
from src.hyperparameter_tuning import tune_hyperparameters, save_model, create_tuning_leaderboard
from src.visualization import (
    plot_logo_cv_folds,
    plot_model_comparison,
    plot_selection_error_distribution,
    create_evaluation_dashboard
)
from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    DEFAULT_MODEL,
    VERBOSE
)


class NECPipeline:
    """
    Complete NEC ML Pipeline.
    
    Integrates all components:
    - Data ingestion and validation
    - Preprocessing and feature engineering
    - Model training with LOGO CV
    - Hyperparameter tuning
    - Comprehensive evaluation
    - Visualization and reporting
    
    Reference: Assessment Brief - "End-to-end ML pipeline"
    """
    
    def __init__(self, verbose=VERBOSE):
        """Initialize pipeline."""
        self.verbose = verbose
        self.train_df = None
        self.test_df = None
        self.preprocessor = None
        self.baseline_model = None
        self.tuned_model = None
        self.baseline_results = None
        self.tuned_results = None
        self.evaluator = ModelEvaluator(verbose=verbose)
        
    def load_data(self):
        """Load and validate data"""
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 1: DATA LOADING")
            print("="*70)
        
        self.train_df, self.test_df = create_train_test(verbose=self.verbose)
        
        if self.verbose:
            print(f"\n Data loaded:")
            print(f"  Training: {self.train_df.shape}")
            print(f"  Testing: {self.test_df.shape}")
    
    def create_preprocessor(self):
        """Create and fit preprocessor"""
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 2: PREPROCESSING")
            print("="*70)
        
        feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        X_train = self.train_df[feature_cols]
        
        preprocessor_obj = NECPreprocessor(verbose=self.verbose)
        preprocessor_obj.fit(X_train)
        self.preprocessor = preprocessor_obj.preprocessor
        
        if self.verbose:
            print(f"\n Preprocessor created and fitted")
    
    def train_baseline(self, model_type=DEFAULT_MODEL):
        """Train baseline model"""
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 3: BASELINE MODEL")
            print("="*70)
        
        # Create baseline pipeline
        self.baseline_model = create_model_pipeline(
            self.preprocessor,
            model_type=model_type,
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"\n Baseline {model_type} model created")
    
    def evaluate_baseline(self, run_logo_cv=True):
        """Evaluate baseline model."""
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 4: BASELINE EVALUATION")
            print("="*70)
        
        self.baseline_results = self.evaluator.evaluate_model_comprehensive(
            self.baseline_model,
            self.train_df,
            self.test_df,
            model_name="Baseline_RF",
            run_logo_cv=run_logo_cv
        )
        
        # Save baseline
        self.evaluator.save_evaluation_report(self.baseline_results, "Baseline_RF")
        self.evaluator.save_selection_table(
            self.baseline_results['test_selection_table'],
            "Baseline_RF",
            "Test"
        )
        
        if self.verbose:
            print(f"\n Baseline evaluation complete")
    
    def tune_model(self, model_type=DEFAULT_MODEL, quick=False):
        """Tune hyperparameters"""
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 5: HYPERPARAMETER TUNING")
            print("="*70)
        
        # Create fresh pipeline for tuning
        pipeline = create_model_pipeline(
            self.preprocessor,
            model_type=model_type,
            verbose=False
        )
        
        # Tune
        tuning_results = tune_hyperparameters(
            pipeline,
            self.train_df,
            model_type=model_type,
            method='grid',
            quick=quick,
            verbose=self.verbose
        )
        
        # Store tuned model
        self.tuned_model = tuning_results['best_estimator']
        
        # Show leaderboard
        if self.verbose:
            print("\n[Tuning Leaderboard - Top 5]")
            print("-"*70)
            leaderboard = create_tuning_leaderboard(tuning_results['cv_results'], top_n=5)
            print(leaderboard[['rank_test_score', 'mean_rmse', 'std_rmse']].to_string(index=False))
        
        # Save tuned model
        save_model(
            self.tuned_model,
            "tuned_rf",
            metadata={
                'best_params': tuning_results['best_params'],
                'best_score': tuning_results['best_score']
            }
        )
        
        if self.verbose:
            print(f"\n Hyperparameter tuning complete")
        
        return tuning_results
    
    def evaluate_tuned(self, run_logo_cv=True):
        """Evaluate tuned model"""
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 6: TUNED MODEL EVALUATION")
            print("="*70)
        
        # Note: tuned_model is already fitted from GridSearchCV
        # We need to wrap it for evaluation
        from sklearn.base import clone
        
        # For evaluation, we pass the already-fitted model
        # But evaluate_model_comprehensive expects unfitted for LOGO CV
        # So we skip LOGO CV here (already done during tuning)
        
        self.tuned_results = self.evaluator.evaluate_model_comprehensive(
            self.tuned_model,
            self.train_df,
            self.test_df,
            model_name="Tuned_RF",
            run_logo_cv=False  # Already tuned with LOGO CV
        )
        
        # Save tuned evaluation
        self.evaluator.save_evaluation_report(self.tuned_results, "Tuned_RF")
        self.evaluator.save_selection_table(
            self.tuned_results['test_selection_table'],
            "Tuned_RF",
            "Test"
        )
        
        if self.verbose:
            print(f"\n Tuned model evaluation complete")
    
    def compare_models(self):
        """Compare baseline vs tuned"""
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 7: MODEL COMPARISON")
            print("="*70)
        
        results_dict = {
            'Baseline_RF': self.baseline_results,
            'Tuned_RF': self.tuned_results
        }
        
        comparison_df = self.evaluator.compare_models(results_dict)
        
        # Create comparison plot
        plot_model_comparison(comparison_df, save=True, show=False)
        
        if self.verbose:
            print(f"\n Model comparison complete")
        
        return comparison_df
    
    def create_visualizations(self):
        """Generate all visualizations"""
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 8: VISUALIZATION")
            print("="*70)
        
        # Baseline visualizations
        if self.baseline_results and self.baseline_results.get('logo_cv'):
            plot_logo_cv_folds(
                self.baseline_results['logo_cv'],
                "Baseline_RF",
                save=True,
                show=False
            )
        
        plot_selection_error_distribution(
            self.baseline_results['test_selection_table'],
            "Baseline_RF",
            save=True,
            show=False
        )
        
        create_evaluation_dashboard(
            self.baseline_results,
            "Baseline_RF",
            save=True,
            show=False
        )
        
        # Tuned visualizations
        plot_selection_error_distribution(
            self.tuned_results['test_selection_table'],
            "Tuned_RF",
            save=True,
            show=False
        )
        
        create_evaluation_dashboard(
            self.tuned_results,
            "Tuned_RF",
            save=True,
            show=False
        )
        
        if self.verbose:
            print(f"\n All visualizations generated")
    
    def run_complete_pipeline(self, tune=True, quick=False):
        """
        Run complete end-to-end pipeline.
        
        Parameters:
        -----------
        tune : bool
            Whether to run hyperparameter tuning
        quick : bool
            Use reduced parameter grid (faster)
        
        Returns:
        --------
        dict : Complete results
        """
        if self.verbose:
            print("\n" + "="*70)
            print("NEC ML PIPELINE - COMPLETE EXECUTION")
            print("="*70)
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Create preprocessor
        self.create_preprocessor()
        
        # Step 3: Train baseline
        self.train_baseline()
        
        # Step 4: Evaluate baseline
        self.evaluate_baseline(run_logo_cv=True)
        
        # Step 5-6: Tune and evaluate (if requested)
        if tune:
            self.tune_model(quick=quick)
            self.evaluate_tuned(run_logo_cv=False)
            
            # Step 7: Compare models
            comparison_df = self.compare_models()
        else:
            comparison_df = None
        
        # Step 8: Create visualizations
        self.create_visualizations()
        
        # Summary
        if self.verbose:
            print("\n" + "="*70)
            print("PIPELINE EXECUTION COMPLETE")
            print("="*70)
            print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\n[FINAL RESULTS SUMMARY]")
            print("-"*70)
            
            if self.baseline_results:
                print(f"\nBaseline Model:")
                print(f"  Test RMSE: ${self.baseline_results['test']['rmse']:.2f}")
                print(f"  Test Selection Error: {self.baseline_results['test']['selection_error']:.2%}")
            
            if self.tuned_results:
                print(f"\nTuned Model:")
                print(f"  Test RMSE: ${self.tuned_results['test']['rmse']:.2f}")
                print(f"  Test Selection Error: {self.tuned_results['test']['selection_error']:.2%}")
                
                # Calculate improvement
                baseline_error = self.baseline_results['test']['selection_error']
                tuned_error = self.tuned_results['test']['selection_error']
                improvement = (baseline_error - tuned_error) / baseline_error * 100
                
                print(f"\nImprovement: {improvement:.1f}%")
            
            print("\n" + "="*70)
        
        return {
            'baseline_results': self.baseline_results,
            'tuned_results': self.tuned_results,
            'comparison': comparison_df
        }


# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING FINAL PIPELINE")
    print("="*70)
    
    # Create pipeline
    pipeline = NECPipeline(verbose=True)
    
    # Run complete pipeline (quick mode for testing)
    results = pipeline.run_complete_pipeline(tune=True, quick=True)
    
    print("\n FINAL PIPELINE TEST PASSED\n")