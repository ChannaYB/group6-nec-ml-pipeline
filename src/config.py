
import os

# FILE PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VALIDATION_REPORTS_DIR = os.path.join(BASE_DIR, 'data', 'validation_reports')

# Raw data files (the 3 files you uploaded)
DEMAND_FILE = 'demand.csv'
PLANTS_FILE = 'plants.csv'
GENERATION_COSTS_FILE = 'generation_costs.csv'

# Processed data files (after merging)
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

# Full paths
DEMAND_PATH = os.path.join(RAW_DATA_DIR, DEMAND_FILE)
PLANTS_PATH = os.path.join(RAW_DATA_DIR, PLANTS_FILE)
GENERATION_COSTS_PATH = os.path.join(RAW_DATA_DIR, GENERATION_COSTS_FILE)

TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, TRAIN_FILE)
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)


# DATA SCHEMA

# Identifier columns
ID_COLUMNS = ['Demand ID', 'Plant ID']

# Target variable (what we're predicting)
TARGET_COLUMN = 'Cost_USD_per_MWh'

# Demand features (from demand.csv)
DEMAND_CATEGORICAL_FEATURES = ['DF_region', 'DF_daytype']
DEMAND_NUMERICAL_FEATURES = [
    'DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'DF6',
    'DF7', 'DF8', 'DF9', 'DF10', 'DF11', 'DF12'
]

# Plant features (from plants.csv)
PLANT_CATEGORICAL_FEATURES = ['Plant Type', 'Region']
PLANT_NUMERICAL_FEATURES = [
    'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6',
    'PF7', 'PF8', 'PF9', 'PF10', 'PF11', 'PF12',
    'PF13', 'PF14', 'PF15', 'PF16', 'PF17', 'PF18'
]

# All features combined
CATEGORICAL_FEATURES = DEMAND_CATEGORICAL_FEATURES + PLANT_CATEGORICAL_FEATURES
NUMERICAL_FEATURES = DEMAND_NUMERICAL_FEATURES + PLANT_NUMERICAL_FEATURES
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# All columns in merged dataset
EXPECTED_COLUMNS = ID_COLUMNS + [TARGET_COLUMN] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Columns that cannot have missing values
MANDATORY_COLUMNS = ID_COLUMNS + [TARGET_COLUMN]

# DATASET CHARACTERISTICS

NUM_DEMAND_SCENARIOS = 500  # Total unique demands
NUM_PLANTS = 64             # Total number of plants
PLANTS_PER_DEMAND = 64      # Each demand has 64 plant options


# VALIDATION SETTINGS

MAX_MISSING_PERCENTAGE = 5.0  # Maximum 5% missing values allowed
MIN_PLANTS_PER_DEMAND = 10    # Minimum plants needed per demand

# Group column for LOGO CV (Leave-One-Group-Out Cross Validation)
GROUP_COLUMN = 'Demand ID'

# TRAIN/TEST SPLIT SETTINGS

RANDOM_SEED = 42                    # For reproducibility
TRAIN_TEST_SPLIT_RATIO = 0.8        # 80% train, 20% test


# LOGGING SETTINGS

VERBOSE = True                      # Show detailed logs
SAVE_VALIDATION_REPORTS = True      # Save validation reports to file

TEST_SIZE = 1 - TRAIN_TEST_SPLIT_RATIO  # 0.2 (20% test)
RANDOM_STATE = RANDOM_SEED 

#preprocessing settings
NUMERICAL_IMPUTATION_STRATEGY = 'median'  # 'mean', 'median', 'most_frequent'
CATEGORICAL_IMPUTATION_STRATEGY = 'most_frequent'  # 'most_frequent', 'constant'

# Scaling method
NUMERICAL_SCALER = 'standard'  # 'standard', 'minmax', 'robust'

# Encoding method
CATEGORICAL_ENCODING = 'onehot'  # 'onehot', 'ordinal'

# Feature engineering
CREATE_INTERACTION_FEATURES = False

# CREATE DIRECTORIES

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(VALIDATION_REPORTS_DIR, exist_ok=True)

# Model selection(step 3)
DEFAULT_MODEL = 'random_forest'  # 'random_forest', 'gradient_boosting', 'linear'
ALTERNATIVE_MODEL = 'gradient_boosting'

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE = RANDOM_SEED

# Gradient Boosting parameters
GB_N_ESTIMATORS = 100
GB_MAX_DEPTH = 5
GB_LEARNING_RATE = 0.1
GB_MIN_SAMPLES_SPLIT = 5
GB_MIN_SAMPLES_LEAF = 2
GB_RANDOM_STATE = RANDOM_SEED

# Linear Regression parameters (baseline)
LINEAR_FIT_INTERCEPT = True

# Number of plants per demand (for selection error calculation)
PLANTS_PER_DEMAND = 64

# EVALUATION SETTINGS (step 4)
# Cross-validation
LOGO_CV_SPLITS = 5  # Number of folds for LOGO CV
LOGO_CV_RANDOM_STATE = RANDOM_SEED

# Evaluation outputs
RESULTS_DIR = 'results'
EVALUATION_REPORTS_DIR = os.path.join(RESULTS_DIR, 'evaluation_reports')
SELECTION_TABLES_DIR = os.path.join(RESULTS_DIR, 'selection_tables')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Create directories if they don't exist
os.makedirs(EVALUATION_REPORTS_DIR, exist_ok=True)
os.makedirs(SELECTION_TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_FIGURE_SIZE = (12, 6)
PLOT_DPI = 100


# HYPERPARAMETER TUNING SETTINGS (Step 5)

# Tuning method
TUNING_METHOD = 'grid'  # 'grid' or 'random'

# Search settings
TUNING_CV_FOLDS = 3  # Use 3 folds for faster tuning (vs 5 for final eval)
TUNING_N_ITER = 20   # For RandomizedSearchCV
TUNING_N_JOBS = -1   # Use all CPU cores
TUNING_VERBOSE = 2   # Progress output

# Random Forest parameter grid
RF_PARAM_GRID = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 20, 30, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2']
}

# Gradient Boosting parameter grid
GB_PARAM_GRID = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# For quick testing (reduced grid)
RF_PARAM_GRID_QUICK = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [10, 20],
    'model__min_samples_split': [2, 5]
}

GB_PARAM_GRID_QUICK = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.1, 0.2]
}

# Model artifacts directory
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)