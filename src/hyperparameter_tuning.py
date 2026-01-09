"""
Hyperparameter Tuning Module for NEC ML Pipeline
Grid/Random search with LOGO CV and custom scorer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

from src.custom_scorer import calculate_selection_error
from src.data_splitting import get_logo_splits
from src.config import (
    TUNING_METHOD,
    TUNING_CV_FOLDS,
    TUNING_N_ITER,
    TUNING_N_JOBS,
    TUNING_VERBOSE,
    RF_PARAM_GRID,
    GB_PARAM_GRID,
    RF_PARAM_GRID_QUICK,
    GB_PARAM_GRID_QUICK,
    MODELS_DIR,
    GROUP_COLUMN,
    VERBOSE
)


def create_logo_cv_splitter(train_df, n_splits=TUNING_CV_FOLDS):
    """
    Create LOGO CV splitter for hyperparameter tuning.
    
    Returns generator of (train_idx, val_idx) tuples.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data
    n_splits : int
        Number of CV folds
    
    Returns:
    --------
    generator : CV splits
    """
    splits = get_logo_splits(train_df, n_splits=n_splits)
    return splits


def create_custom_scorer_for_tuning():
    """
    Create custom scorer for hyperparameter tuning.

    The scorer must work with sklearn's CV framework.
    Returns NEGATIVE RMSE (sklearn maximizes scores).

    Note: We use RMSE as a proxy during hyperparameter tuning since
    selection error requires demand_ids which are not easily passed
    through GridSearchCV. The final evaluation uses selection error.

    Returns:
    --------
    scorer : sklearn scorer object

    Reference: Assessment Brief - "Custom selection-error scorer"
    """
    def rmse_metric(y_true, y_pred):
        """
        Calculate RMSE metric for hyperparameter tuning.

        This is a simple metric function compatible with make_scorer.
        """
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    # make_scorer expects a metric function (y_true, y_pred) not (estimator, X, y)
    # greater_is_better=False means we want to minimize RMSE
    scorer = make_scorer(rmse_metric, greater_is_better=False)
    return scorer


def tune_hyperparameters(
    pipeline,
    train_df,
    model_type='random_forest',
    method='grid',
    param_grid=None,
    quick=False,
    verbose=VERBOSE
):
    """
    Tune hyperparameters using Grid or Random Search with LOGO CV.
    
    Parameters:
    -----------
    pipeline : sklearn Pipeline
        Unfitted pipeline (preprocessor + model)
    train_df : pandas.DataFrame
        Training data with all columns
    model_type : str
        'random_forest' or 'gradient_boosting'
    method : str
        'grid' or 'random'
    param_grid : dict, optional
        Custom parameter grid
    quick : bool
        Use reduced grid for faster testing
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict : Tuning results containing best model, params, and scores
    
    Reference: Assessment Brief - "Hyper-parameter optimisation (evidence-led)"
    """
    if verbose:
        print("\n" + "="*70)
        print(f"HYPERPARAMETER TUNING: {model_type.upper()}")
        print("="*70)
    
    from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
    
    # Prepare data
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    groups = train_df[GROUP_COLUMN]
    
    # Get parameter grid
    if param_grid is None:
        if model_type == 'random_forest':
            param_grid = RF_PARAM_GRID_QUICK if quick else RF_PARAM_GRID
        elif model_type == 'gradient_boosting':
            param_grid = GB_PARAM_GRID_QUICK if quick else GB_PARAM_GRID
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Create LOGO CV splits
    cv_splits = list(get_logo_splits(train_df, n_splits=TUNING_CV_FOLDS))
    
    if verbose:
        print(f"\n[1] Configuration")
        print(f"  Method: {method.upper()} Search")
        print(f"  CV Folds: {len(cv_splits)} (LOGO)")
        print(f"  Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
        print(f"  Quick mode: {quick}")
    
    # Create scorer (using RMSE as proxy for selection error)
    scorer = create_custom_scorer_for_tuning()
    
    # Create search object
    if method == 'grid':
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_splits,
            scoring=scorer,
            n_jobs=TUNING_N_JOBS,
            verbose=TUNING_VERBOSE,
            refit=True,
            return_train_score=True
        )
    else:  # random
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=TUNING_N_ITER,
            cv=cv_splits,
            scoring=scorer,
            n_jobs=TUNING_N_JOBS,
            verbose=TUNING_VERBOSE,
            random_state=42,
            refit=True,
            return_train_score=True
        )
    
    # Run search
    if verbose:
        print(f"\n[2] Running {method.upper()} Search...")
        print(f"  This may take several minutes...")
    
    search.fit(X_train, y_train, groups=groups)
    
    # Extract results
    results = {
        'best_estimator': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_),
        'search_object': search
    }
    
    if verbose:
        print(f"\n[3] Tuning Complete")
        print(f"  Best score (neg RMSE): {search.best_score_:.4f}")
        print(f"  Best RMSE: ${abs(search.best_score_):.2f}")
        print(f"\n  Best parameters:")
        for param, value in search.best_params_.items():
            print(f"    {param}: {value}")
    
    return results


def create_tuning_leaderboard(cv_results_df, top_n=10):
    """
    Create leaderboard of top parameter combinations.
    
    Parameters:
    -----------
    cv_results_df : pandas.DataFrame
        Results from GridSearchCV/RandomizedSearchCV
    top_n : int
        Number of top results to show
    
    Returns:
    --------
    pandas.DataFrame : Leaderboard
    
    Reference: Assessment Brief - "Tuning leaderboard"
    """
    # Select relevant columns
    cols = ['mean_test_score', 'std_test_score', 'rank_test_score']
    param_cols = [c for c in cv_results_df.columns if c.startswith('param_')]

    leaderboard = cv_results_df[cols + param_cols].copy()

    # Convert scores to RMSE (scores are negative, we want positive RMSE values)
    leaderboard['mean_rmse'] = abs(leaderboard['mean_test_score'])
    leaderboard['std_rmse'] = leaderboard['std_test_score']
    
    # Sort by rank
    leaderboard = leaderboard.sort_values('rank_test_score')
    
    # Return top N
    return leaderboard.head(top_n)


def save_model(model, model_name, metadata=None):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn estimator
        Fitted model
    model_name : str
        Name for the model file
    metadata : dict, optional
        Additional metadata to save
    
    Returns:
    --------
    str : Path to saved model
    
    Reference: Assessment Brief - "Saved artefacts"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = Path(MODELS_DIR) / filename
    
    # Create save object
    save_obj = {
        'model': model,
        'model_name': model_name,
        'timestamp': timestamp,
        'metadata': metadata or {}
    }
    
    # Save
    with open(filepath, 'wb') as f:
        pickle.dump(save_obj, f)
    
    print(f" Model saved: {filepath}")
    
    return str(filepath)


def load_model(filepath):
    """
    Load saved model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to saved model
    
    Returns:
    --------
    dict : Loaded model object with metadata
    """
    with open(filepath, 'rb') as f:
        save_obj = pickle.load(f)
    
    return save_obj


# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING HYPERPARAMETER TUNING MODULE")
    print("="*70)
    
    try:
        # Load data
        from src.data_ingestion import create_train_test
        from src.preprocessing import NECPreprocessor
        from src.models import create_model_pipeline
        from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
        
        print("\n[1] Loading data...")
        train_df, test_df = create_train_test(verbose=False)
        print(f" Data loaded: Train {train_df.shape}, Test {test_df.shape}")
        
        # Create pipeline
        print("\n[2] Creating pipeline...")
        feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        X_train = train_df[feature_cols]
        
        preprocessor_obj = NECPreprocessor(verbose=False)
        preprocessor_obj.fit(X_train)
        
        pipeline = create_model_pipeline(
            preprocessor_obj.preprocessor,
            model_type='random_forest',
            verbose=False
        )
        print(" Pipeline created")
        
        # Tune (quick mode for testing)
        print("\n[3] Running hyperparameter tuning (quick mode)...")
        tuning_results = tune_hyperparameters(
            pipeline,
            train_df,
            model_type='random_forest',
            method='grid',
            quick=True,  # Use reduced grid
            verbose=True
        )
        
        # Show leaderboard
        print("\n[4] Top 5 Parameter Combinations:")
        print("-"*70)
        leaderboard = create_tuning_leaderboard(tuning_results['cv_results'], top_n=5)
        print(leaderboard[['rank_test_score', 'mean_rmse', 'std_rmse']].to_string(index=False))
        
        # Save model
        print("\n[5] Saving best model...")
        model_path = save_model(
            tuning_results['best_estimator'],
            "test_tuned_rf",
            metadata={'best_params': tuning_results['best_params']}
        )
        
        print("\n" + "="*70)
        print(" HYPERPARAMETER TUNING MODULE TEST PASSED")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        import traceback
        traceback.print_exc()