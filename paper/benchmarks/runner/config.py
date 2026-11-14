# config.py
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    RandomForestRegressor, ExtraTreesRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# ==========================================
# 1. EXPERIMENT SETTINGS
# ==========================================
N_RUNS = 10
RANDOM_STATE = 42
VERSION = "article_fold_selection"  # Keep corrected scores separate from preselection-CV results.
RESULTS_FILE = F"../data/benchmark_results_{VERSION}.csv"
CHECKPOINT_RESULTS_FILE = F"../data/checkpoint_history_vanilla_{VERSION}.csv"
SYNTHETIC_GT_RESULTS_FILE = F"../data/benchmark_gt_results_{VERSION}.csv"
ALPHAS = [1e-1, 1e-3, 1e-5]
# ALPHAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

# ==========================================
# 2. DATASETS (OpenML IDs or synthetic configs)
# ==========================================

# Synthetic dataset configs (passed to load_synthetic_dataset)
CORRELATED_NOISE_CONFIG_REDUNDANT_LARGE = {
    'n_samples': 1000,
    'n_features': 500,
    'n_informative': 5,
    'n_redundant': 100,  # LOTS of redundant features correlated with informative
    'n_repeated': 0,
    'n_clusters_per_class': 1,
    'class_sep': 0.0001,
    'flip_y': 0.3,  # Some label noise
    'random_state': 42,
    'dataset_name': 'correlated_noise',
    'shuffle': False,
}

DATASETS = {
    "Madelon": 1485,      # Synthetic, Non-linear, Binary Class
    "Arcene": 1458,       # Mass Spec, P >> N, Binary Class
    "Gisette": 41026,     # Handwriting, 5000 feats (NIPS challenge)
    "Dexter": 4136,      # Text classification, 20,000 feats (sparse)
    "riboflavin": 46983, # Gene Expression, Regression (10k features)
    # "Superconduct": 42224, # Physics, Regression (Proxy for Riboflavin)
    "Isolet": 300,        # Speech Rec, Multi-class (26 classes)
    # "Dorothea": 4137,       # Drug Discovery, 100k feats, Binary Class
    # "correlated_noise": CORRELATED_NOISE_CONFIG_REDUNDANT_LARGE,  # Synthetic, high noise
}

# Map dataset names to task type: 'class' or 'reg'
TASK_MAP = {
    "Madelon": "class",
    "Arcene": "class",
    "Gisette": "class",
    "Dexter": "class",
    "Dorothea": "class",
    "Isolet": "class",
    "Superconduct": "reg",
    "riboflavin": "reg",
    "correlated_noise": "class",
}


# ==========================================
# 3. ESTIMATORS TO TEST
# ==========================================
# You can comment out heavy ones if needed
# ESTIMATORS_LIST = ["extra_trees", "rf", "lgbm", "xgb"]
# ESTIMATORS_LIST = ["extra_trees", "rf"]
ESTIMATORS_LIST = ["extra_trees"]
# ESTIMATORS_LIST = ["rf"]

def get_base_estimator(task_type="class", model_type="extra_trees", n_jobs=-1):
    """
    Returns the base estimator used INSIDE Boruta.
    """
    # --- EXTRA TREES ---
    if model_type == "extra_trees":
        if task_type == "class":
            return ExtraTreesClassifier(max_depth=5, n_jobs=n_jobs, random_state=RANDOM_STATE)
        return ExtraTreesRegressor(max_depth=5, n_jobs=n_jobs, random_state=RANDOM_STATE)

    # --- RANDOM FOREST ---
    elif model_type == "rf":
        if task_type == "class":
            return RandomForestClassifier(max_depth=5, n_jobs=n_jobs, class_weight='balanced', random_state=RANDOM_STATE)
        return RandomForestRegressor(max_depth=5, n_jobs=n_jobs, random_state=RANDOM_STATE)

    # --- LIGHTGBM ---
    elif model_type == "lgbm":
        if task_type == "class":
            return LGBMClassifier(
                max_depth=5, n_estimators=500, n_jobs=n_jobs, 
                random_state=RANDOM_STATE, importance_type='gain', verbose=-1
            )
        return LGBMRegressor(
            max_depth=5, n_estimators=500, n_jobs=n_jobs, 
            random_state=RANDOM_STATE, importance_type='gain', verbose=-1
        )

    # --- XGBOOST ---
    elif model_type == "xgb":
        if task_type == "class":
            return XGBClassifier(
                max_depth=5, n_estimators=500, n_jobs=n_jobs, 
                random_state=RANDOM_STATE, importance_type='gain', verbosity=0,
                tree_method="hist" # Faster on big data
            )
        return XGBRegressor(
            max_depth=5, n_estimators=500, n_jobs=n_jobs, 
            random_state=RANDOM_STATE, importance_type='gain', verbosity=0,
            tree_method="hist"
        )

    raise ValueError(f"Unknown model/task type: {model_type}/{task_type}")

# ==========================================
# 4. BORUTA PARAMETERS
# ==========================================
# ALPHAS = [0.05, 0.01, 0.001, 1e-5, 1e-7]


###########################################################

# from sklearn.ensemble import (
#     RandomForestClassifier, ExtraTreesClassifier,
#     RandomForestRegressor, ExtraTreesRegressor
# )

# # ==========================================
# # 1. EXPERIMENT SETTINGS
# # ==========================================
# N_RUNS = 1  
# RANDOM_STATE = 42

# # ==========================================
# # 2. DATASETS (OpenML IDs)
# # ==========================================
# DATASETS = {
#     # "Madelon": 1485,      # Synthetic, Non-linear, Binary Class
#     # "Arcene": 1458,       # Mass Spec, P >> N, Binary Class
#     # "Gisette": 41026,     # Handwriting, 5000 feats (NIPS challenge)
#     # "Dexter": 4136,      # Text classification, 20,000 feats (sparse)
#     # "Isolet": 300,        # Speech Rec, Multi-class (26 classes)
#     # "Superconduct": 42224, # Physics, Regression (Proxy for Riboflavin)
#     "riboflavin": 46983, # Gene Expression, Regression (10k features) 
#     # "Dorothea": 4137,       # Drug Discovery, 100k feats, Binary Class
# }

# # Map dataset names to task type: 'class' or 'reg'
# TASK_MAP = {
#     "Madelon": "class",
#     "Arcene": "class",
#     "Gisette": "class",      
#     "Dexter": "class",       
#     "Dorothea": "class",
#     "Isolet": "class",
#     "Superconduct": "reg",
#     "riboflavin": "reg"
# }

# # ==========================================
# # 3. BASE ESTIMATOR FACTORY
# # ==========================================
# def get_base_estimator(task_type="class", model_type="extra_trees", n_jobs=-1):
#     """
#     Returns the base estimator used INSIDE Boruta.
#     Switch between Classifier and Regressor based on task.
#     """
#     if task_type == "class":
#         if model_type == "extra_trees":
#             return ExtraTreesClassifier(
#                 max_depth=5,
#                 n_jobs=n_jobs, 
#                 random_state=RANDOM_STATE
#             )
#         elif model_type == "rf":
#             return RandomForestClassifier(
#                 max_depth=5, 
#                 n_jobs=n_jobs, 
#                 class_weight='balanced',
#                 random_state=RANDOM_STATE
#             )
#     elif task_type == "reg":
#         if model_type == "extra_trees":
#             return ExtraTreesRegressor(
#                 max_depth=5,
#                 n_jobs=n_jobs, 
#                 random_state=RANDOM_STATE
#             )
#         elif model_type == "rf":
#             return RandomForestRegressor(
#                 max_depth=5, 
#                 n_jobs=n_jobs, 
#                 random_state=RANDOM_STATE
#             )
            
#     raise ValueError(f"Unknown model/task type: {model_type}/{task_type}")

# # ==========================================
# # 4. BORUTA PARAMETERS
# # ==========================================
# ALPHAS = [0.05, 0.01, 0.001, 1e-5, 1e-7]


#########################################################

# # config.py
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# # import xgboost as xgb # Uncomment later
# # import lightgbm as lgbm # Uncomment later

# # ==========================================
# # 1. EXPERIMENT SETTINGS
# # ==========================================
# N_RUNS = 1  # How many times to repeat for stability stats
# RANDOM_STATE = 42
# # TEST_SIZE = 0.3  # Hold-out set for downstream validation

# # ==========================================
# # 2. DATASETS (OpenML IDs)
# # ==========================================
# # We target high-dimensional datasets often used in FS literature
# DATASETS = {
#     "Madelon": 1485,      # Synthetic, 500 feats, highly non-linear
#     "Arcene": 1458,       # Mass Spec, 10,000 feats (Classic NIPS FS challenge)
#     # "APSFailure": 41138,     # Scania car dataset
#     # "Helena": 41169,      # AutoML Challenge
#     # "Gisette": 41026,     # Handwriting, 5000 feats (NIPS challenge)
#     # "Dexter": 4136,      # Text classification, 20,000 feats (sparse)
# }

# # ==========================================
# # 3. BASE ESTIMATOR FACTORY
# # ==========================================
# def get_base_estimator(model_type="extra_trees", n_jobs=-1):
#     """
#     Returns the base estimator used INSIDE Boruta.
#     Standard Boruta papers use RF or ExtraTrees.
#     """
#     if model_type == "extra_trees":
#         return ExtraTreesClassifier(
#             # n_estimators=1000, 
#             max_depth=4,        # Constrained depth helps speed/overfitting
#             n_jobs=n_jobs, 
#             random_state=RANDOM_STATE
#         )
#     elif model_type == "rf":
#         return RandomForestClassifier(
#             # n_estimators=200, 
#             max_depth=5, 
#             n_jobs=n_jobs, 
#             class_weight='balanced',
#             random_state=RANDOM_STATE
#         )
#     # Add LGBM/XGB here later
#     raise ValueError(f"Unknown model type: {model_type}")

# # ==========================================
# # 4. BORUTA PARAMETERS
# # ==========================================
# # The alpha values to test.
# ALPHAS = [0.05, 0.01, 0.001, 1e-5]
