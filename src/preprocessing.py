"""
Preprocessing Module for NEC ML Pipeline
Handles feature transformation, scaling, and encoding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ID_COLUMNS,
    TARGET_COLUMN,
    NUMERICAL_IMPUTATION_STRATEGY,
    CATEGORICAL_IMPUTATION_STRATEGY,
    NUMERICAL_SCALER,
    CATEGORICAL_ENCODING,
    VERBOSE
)


class NECPreprocessor: 
    """
    Preprocessing pipeline for NEC dataset.
    
    Handles:
    - Missing value imputation
    - Numerical feature scaling
    - Categorical feature encoding
    """   
    def __init__(self, verbose=VERBOSE):
        """Initialize preprocessor."""
        self.verbose = verbose
        self.preprocessor = None
        self.feature_names_out = None
        
    def _create_numerical_pipeline(self):
        """
        Create pipeline for numerical features.
        
        Steps:
        1. Impute missing values (median by default)
        2. Scale features (StandardScaler by default)
        
        Returns:
        --------
        Pipeline : Numerical transformation pipeline
        """
        steps = []
        
        # Step 1: Imputation
        imputer = SimpleImputer(strategy=NUMERICAL_IMPUTATION_STRATEGY)
        steps.append(('imputer', imputer))
        
        # Step 2: Scaling
        if NUMERICAL_SCALER == 'standard':
            scaler = StandardScaler()
        elif NUMERICAL_SCALER == 'minmax':
            scaler = MinMaxScaler()
        elif NUMERICAL_SCALER == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()  # Default
        
        steps.append(('scaler', scaler))
        
        return Pipeline(steps)
    
    def _create_categorical_pipeline(self):
        steps = []
        
        # Step 1: Imputation
        imputer = SimpleImputer(strategy=CATEGORICAL_IMPUTATION_STRATEGY)
        steps.append(('imputer', imputer))
        
        # Step 2: Encoding
        if CATEGORICAL_ENCODING == 'onehot':
            encoder = OneHotEncoder(
                drop='first',  # Avoid multicollinearity
                sparse_output=False,
                handle_unknown='ignore'
            )
            steps.append(('encoder', encoder))
        
        return Pipeline(steps)
    
    def build_preprocessor(self):
        """
        Build complete preprocessing pipeline.
        
        Uses ColumnTransformer to apply different transformations
        to numerical and categorical features.
        
        Returns:
        --------
        ColumnTransformer : Complete preprocessing pipeline
        
        Reference: Assessment Brief - "unified scikit-learn Pipeline"
        """
        if self.verbose:
            print(" Building preprocessing pipeline...")
            print(f"  Numerical features: {len(NUMERICAL_FEATURES)}")
            print(f"  Categorical features: {len(CATEGORICAL_FEATURES)}")
        
        # Create transformers
        numerical_pipeline = self._create_numerical_pipeline()
        categorical_pipeline = self._create_categorical_pipeline()
        
        # Combine into ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, NUMERICAL_FEATURES),
                ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
            ],
            remainder='drop',  # Drop ID columns and target
            verbose=self.verbose
        )
        
        self.preprocessor = preprocessor
        
        if self.verbose:
            print(f" Preprocessing pipeline built")
            print(f"  - Numerical: {NUMERICAL_IMPUTATION_STRATEGY} imputation → {NUMERICAL_SCALER} scaling")
            print(f"  - Categorical: {CATEGORICAL_IMPUTATION_STRATEGY} imputation → {CATEGORICAL_ENCODING} encoding")
        
        return preprocessor
    
    def fit(self, X_train, y_train=None):
        """
        Fit preprocessor on training data.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features (must include all feature columns)
        y_train : pandas.Series, optional
            Training target (not used, included for sklearn compatibility)
        
        Returns:
        --------
        self : NECPreprocessor
            Fitted preprocessor
        """
        if self.preprocessor is None:
            self.build_preprocessor()
        
        if self.verbose:
            print(f"\nFitting preprocessor on training data...")
            print(f"  Training shape: {X_train.shape}")
        
        # Fit on training data
        self.preprocessor.fit(X_train)
        
        # Get feature names after transformation
        try:
            self.feature_names_out = self.preprocessor.get_feature_names_out()
        except:
            # Fallback if get_feature_names_out not available
            self.feature_names_out = None
        
        if self.verbose:
            print(f" Preprocessor fitted")
            if self.feature_names_out is not None:
                print(f"  Output features: {len(self.feature_names_out)}")
        
        return self
    
    def transform(self, X):
        """
        Transform features using fitted preprocessor.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features to transform
        
        Returns:
        --------
        numpy.ndarray : Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted! Call fit() first.")
        
        if self.verbose:
            print(f"\nTransforming data...")
            print(f"  Input shape: {X.shape}")
        
        X_transformed = self.preprocessor.transform(X)
        
        if self.verbose:
            print(f" Data transformed")
            print(f"  Output shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X_train, y_train=None):
        """
        Fit preprocessor and transform training data.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series, optional
            Training target
        
        Returns:
        --------
        numpy.ndarray : Transformed training features
        """
        self.fit(X_train, y_train)
        return self.transform(X_train)

# CONVENIENCE FUNCTIONS

def preprocess_data(train_df, test_df, verbose=VERBOSE):
    """
    Complete preprocessing workflow.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data (with all columns)
    test_df : pandas.DataFrame
        Testing data (with all columns)
    verbose : bool
        Print progress
    
    Returns:
    --------
    tuple : (X_train_transformed, X_test_transformed, y_train, y_test, preprocessor)
    
    """
    if verbose:
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE")
        print("="*70)
    
    # Separate features and target
    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]
    
    if verbose:
        print(f"\n[1] Data Separation")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
    
    # Create and fit preprocessor
    if verbose:
        print(f"\n[2] Building Preprocessor")
    
    preprocessor = NECPreprocessor(verbose=verbose)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    if verbose:
        print(f"\n[3] Transformation Complete")
        print(f"  X_train_transformed: {X_train_transformed.shape}")
        print(f"  X_test_transformed: {X_test_transformed.shape}")
        print("\n" + "="*70)
        print(" PREPROCESSING COMPLETE")
        print("="*70 + "\n")
    
    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor


# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PREPROCESSING MODULE")
    print("="*70)
    
    try:
        # Load data
        from src.data_ingestion import create_train_test
        
        print("\n[1] Loading data")
        train_df, test_df = create_train_test(verbose=False)
        print(f" Data loaded: Train {train_df.shape}, Test {test_df.shape}")
        
        # Run preprocessing
        print("\n[2] Running preprocessing pipeline...")
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            train_df, test_df, verbose=True
        )
        
        # Verification
        print("\n" + "="*70)
        print("VERIFICATION")
        print("="*70)
        
        print(f"\n Shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
        print(f"\n No NaN in transformed data:")
        print(f"  X_train NaN: {np.isnan(X_train).sum()}")
        print(f"  X_test NaN: {np.isnan(X_test).sum()}")
        
        print(f"\n Data type: {type(X_train)}")
        print(f" Target preserved: {len(y_train)} train, {len(y_test)} test")
        
        print("\n" + "="*70)
        print(" PREPROCESSING MODULE TEST PASSED")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n Error: {e}\n")
        import traceback
        traceback.print_exc()