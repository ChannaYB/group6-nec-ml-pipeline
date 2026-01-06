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
    def __init__(self, verbose=VERBOSE):
        """Initialize preprocessor."""
        self.verbose = verbose
        self.preprocessor = None
        self.feature_names_out = None
        
    def _create_numerical_pipeline(self):
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
        self.fit(X_train, y_train)
        return self.transform(X_train)