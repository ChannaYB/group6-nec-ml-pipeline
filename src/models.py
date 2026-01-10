"""
Model Implementations for NEC ML Pipeline
Provides multiple regression models with LOGO CV support
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from src.config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF, RF_RANDOM_STATE,
    GB_N_ESTIMATORS, GB_MAX_DEPTH, GB_LEARNING_RATE, GB_MIN_SAMPLES_SPLIT, GB_MIN_SAMPLES_LEAF, GB_RANDOM_STATE,
    LINEAR_FIT_INTERCEPT,
    GROUP_COLUMN,
    VERBOSE
)


def create_random_forest_model():
    """
    Create Random Forest Regressor.
    """
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    return model


def create_gradient_boosting_model():
    """
    Create Gradient Boosting Regressor.
    """
    model = GradientBoostingRegressor(
        n_estimators=GB_N_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        min_samples_split=GB_MIN_SAMPLES_SPLIT,
        min_samples_leaf=GB_MIN_SAMPLES_LEAF,
        random_state=GB_RANDOM_STATE,
        verbose=0
    )
    
    return model


def create_linear_regression_model():
    """
    Create Linear Regression model.
    """
    model = LinearRegression(
        fit_intercept=LINEAR_FIT_INTERCEPT
    )
    
    return model


def create_model_pipeline(preprocessor, model_type='random_forest', verbose=VERBOSE):
    """
    Create complete ML pipeline: Preprocessor + Model.
    """
    # Select model
    if model_type == 'random_forest':
        model = create_random_forest_model()
        model_name = "Random Forest"
    elif model_type == 'gradient_boosting':
        model = create_gradient_boosting_model()
        model_name = "Gradient Boosting"
    elif model_type == 'linear':
        model = create_linear_regression_model()
        model_name = "Linear Regression"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    if verbose:
        print(f"ℹ Created pipeline: Preprocessor → {model_name}")
    
    return pipeline


def train_model(pipeline, X_train, y_train, verbose=VERBOSE):
    """
    Train the model pipeline
    """
    if verbose:
        print(f"\nℹ Training model...")
        print(f"  Training samples: {len(X_train):,}")
    
    # Fit pipeline (preprocessor + model)
    pipeline.fit(X_train, y_train)
    
    if verbose:
        print(f" Model trained")
    
    return pipeline


def evaluate_model(pipeline, X, y, demand_ids, plant_ids, dataset_name="Data", verbose=VERBOSE):
    """
    Evaluate model performance
    """
    # Get predictions
    y_pred = pipeline.predict(X)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Calculate R²
    r2 = r2_score(y, y_pred)
    
    # Calculate selection error
    from src.custom_scorer import calculate_selection_error
    selection_error = calculate_selection_error(y, y_pred, demand_ids)
    
    # Store results
    results = {
        'dataset': dataset_name,
        'rmse': rmse,
        'r2': r2,
        'selection_error': selection_error,
        'n_samples': len(y),
        'n_demands': len(np.unique(demand_ids))
    }
    
    if verbose:
        print(f"\n {dataset_name} Results:")
        print(f"  RMSE: ${rmse:.2f}/MWh")
        print(f"  R²: {r2:.4f}")
        print(f"  Selection Error: {selection_error:.2%} ({int(selection_error * results['n_demands'])}/{results['n_demands']} demands)")
    
    return results


def evaluate_with_logo_cv(pipeline, train_df, n_splits=5, verbose=VERBOSE):
    """
    Evaluate model using Leave-One-Group-Out Cross-Validation.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOGO CROSS-VALIDATION ({n_splits} folds)")
        print(f"{'='*70}")
    
    from src.data_splitting import get_logo_splits
    from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
    
    # Get features
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = train_df[feature_cols]
    y = train_df[TARGET_COLUMN]
    demand_ids = train_df[GROUP_COLUMN]
    plant_ids = train_df['Plant ID']
    
    # Get LOGO splits
    splits = get_logo_splits(train_df, n_splits=n_splits)
    
    # Store results for each fold
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print(f"{'─'*70}")
        
        # Split data
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]
        
        demand_ids_val = demand_ids.iloc[val_idx]
        plant_ids_val = plant_ids.iloc[val_idx]
        
        # Clone pipeline (create fresh copy)
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)
        
        # Train on fold
        fold_pipeline.fit(X_fold_train, y_fold_train)
        
        # Evaluate on validation fold
        val_results = evaluate_model(
            fold_pipeline,
            X_fold_val,
            y_fold_val,
            demand_ids_val,
            plant_ids_val,
            dataset_name=f"Fold {fold_idx + 1} Val",
            verbose=verbose
        )
        
        fold_results.append(val_results)
    
    # Aggregate results
    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    std_rmse = np.std([r['rmse'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    avg_selection_error = np.mean([r['selection_error'] for r in fold_results])
    std_selection_error = np.std([r['selection_error'] for r in fold_results])
    
    cv_results = {
        'n_splits': n_splits,
        'fold_results': fold_results,
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'avg_r2': avg_r2,
        'avg_selection_error': avg_selection_error,
        'std_selection_error': std_selection_error
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOGO CV SUMMARY")
        print(f"{'='*70}")
        print(f"Average RMSE: ${avg_rmse:.2f} ± ${std_rmse:.2f}")
        print(f"Average R²: {avg_r2:.4f}")
        print(f"Average Selection Error: {avg_selection_error:.2%} ± {std_selection_error:.2%}")
        print(f"{'='*70}")
    
    return cv_results


# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING MODELS MODULE WITH LOGO CV")
    print("="*70)
    
    try:
        # Load data
        from src.data_ingestion import create_train_test
        from src.preprocessing import NECPreprocessor
        from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
        
        print("\n[1] Loading data...")
        train_df, test_df = create_train_test(verbose=False)
        print(f" Data loaded: Train {train_df.shape}, Test {test_df.shape}")
        
        # Prepare features
        feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        X_train = train_df[feature_cols]
        y_train = train_df['Cost_USD_per_MWh']
        X_test = test_df[feature_cols]
        y_test = test_df['Cost_USD_per_MWh']
        
        # Create preprocessor
        print("\n[2] Creating preprocessor...")
        preprocessor_obj = NECPreprocessor(verbose=False)
        preprocessor_obj.fit(X_train)
        print(" Preprocessor fitted")
        
        # Test Random Forest with LOGO CV
        print(f"\n[3] Testing Random Forest with LOGO CV")
        print("="*70)
        
        # Create pipeline (unfitted)
        pipeline = create_model_pipeline(
            preprocessor_obj.preprocessor,
            model_type='random_forest',
            verbose=True
        )
        
        # LOGO Cross-validation
        cv_results = evaluate_with_logo_cv(
            pipeline,
            train_df,
            n_splits=5,
            verbose=True
        )
        
        # Train on full training set
        print(f"\n[4] Training on Full Training Set")
        print("-"*70)
        final_pipeline = create_model_pipeline(
            preprocessor_obj.preprocessor,
            model_type='random_forest',
            verbose=False
        )
        final_pipeline = train_model(final_pipeline, X_train, y_train, verbose=True)
        
        # Evaluate on test set
        print(f"\n[5] Final Test Set Evaluation")
        print("-"*70)
        test_results = evaluate_model(
            final_pipeline,
            X_test,
            y_test,
            test_df['Demand ID'],
            test_df['Plant ID'],
            dataset_name="Test",
            verbose=True
        )
        
        print("\n" + "="*70)
        print(" MODELS MODULE WITH LOGO CV TEST PASSED")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n Error: {e}\n")
        import traceback
        traceback.print_exc()