import openml
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_synthetic_dataset(synth_config, random_state=42):
    """
    Generates a synthetic classification dataset using sklearn's make_classification.
    Args:
        synth_config: dict with make_classification parameters plus optional 'dataset_name'.
    Returns:
        X (numpy array), y (numpy array), ground_truth (dict with 'strict' and 'broad' index sets)
    """
    cfg = {k: v for k, v in synth_config.items() if k != 'dataset_name'}
    name = synth_config.get('dataset_name', 'synthetic')
    print(f"Generating synthetic dataset '{name}'...")

    X, y = make_classification(**cfg)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # make_classification places features in order: informative, redundant, repeated, noise
    n_informative = cfg.get('n_informative', 2)
    n_redundant = cfg.get('n_redundant', 2)
    ground_truth = {
        'strict': set(range(n_informative)),
        'broad': set(range(n_informative + n_redundant)),
    }

    print(f"Synthetic dataset generated: {X.shape[0]} samples, {X.shape[1]} features. "
          f"GT strict={len(ground_truth['strict'])}, broad={len(ground_truth['broad'])}.")
    return X, y, ground_truth


def load_openml_dataset(dataset_id, task_type="class", preprocess=True):
    """
    Fetches dataset from OpenML.
    Args:
        dataset_id: OpenML ID
        task_type: 'class' or 'reg'. Affects target encoding.
    """
    print(f"Downloading dataset ID {dataset_id} ({task_type})...")
    
    dataset = openml.datasets.get_dataset(dataset_id)
    
    try:
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format='dataframe'
        )
    except Exception:
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
    
    # 1. Sparse to Dense
    if hasattr(X, "sparse"):
        try:
            X = X.sparse.to_dense()
        except:
            pass
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=attribute_names)

    if not preprocess:
        # The publication runner learns imputers/encoders on each training fold.
        if y is None:
            raise ValueError("Target variable is None.")
        y = np.asarray(y)
        valid = ~pd.isna(y)
        X, y = X.loc[valid].reset_index(drop=True), y[valid]
        y = LabelEncoder().fit_transform(y) if task_type == "class" else y.astype(float)
        return X, y

    # 2. Split Numeric/Categorical
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    
    X_num = X[num_cols].values if len(num_cols) > 0 else None
    X_cat = X[cat_cols].values if len(cat_cols) > 0 else None
    
    # 3. Impute
    if X_num is not None:
        try:
            imp_num = SimpleImputer(strategy='mean', keep_empty_features=True)
        except TypeError:
            imp_num = SimpleImputer(strategy='mean')
        X_num = imp_num.fit_transform(X_num)
        
    if X_cat is not None:
        try:
            imp_cat = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
        except TypeError:
            imp_cat = SimpleImputer(strategy='most_frequent')
        X_cat = imp_cat.fit_transform(X_cat)
        
        # Ordinal Encode Categoricals
        X_cat_enc = []
        for i in range(X_cat.shape[1]):
            le = LabelEncoder()
            col_data = X_cat[:, i].astype(str)
            X_cat_enc.append(le.fit_transform(col_data))
        
        if len(X_cat_enc) > 0:
            X_cat = np.column_stack(X_cat_enc)
        else:
            X_cat = None

    # 4. Concatenate
    if X_num is not None and X_cat is not None:
        X_final = np.column_stack([X_num, X_cat])
    elif X_num is not None:
        X_final = X_num
    elif X_cat is not None:
        X_final = X_cat
    else:
        raise ValueError("Dataset resulted in 0 features.")
        
    # 5. Target Processing
    if y is None:
        raise ValueError("Target variable is None.")
    
    y = np.array(y)
    
    if task_type == "class":
        # Encode labels for classification
        y_final = LabelEncoder().fit_transform(y)
    else:
        # Keep as float for regression
        y_final = y.astype(float)
        # Check for NaNs in target (rare but possible in OpenML)
        if np.isnan(y_final).any():
            # Drop rows with NaN targets (simple approach)
            mask = ~np.isnan(y_final)
            X_final = X_final[mask]
            y_final = y_final[mask]

    # 6. Scale Features
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_final)
    
    print(f"Dataset loaded: {X_final.shape[0]} samples, {X_final.shape[1]} features.")
    return X_final, y_final

#########################################################

# # data_loader.py
# import openml
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.impute import SimpleImputer

# def load_openml_dataset(dataset_id):
#     """
#     Fetches dataset from OpenML, handles NaN and encoding.
#     Returns: X (numpy array), y (numpy array)
#     """
#     print(f"Downloading dataset ID {dataset_id}...")
    
#     # Fetch dataset with 'dataframe' format to inspect types easily
#     dataset = openml.datasets.get_dataset(dataset_id)
    
#     try:
#         # OpenML python API can sometimes return sparse frames
#         X, y, categorical_indicator, attribute_names = dataset.get_data(
#             target=dataset.default_target_attribute,
#             dataset_format='dataframe'
#         )
#     except Exception:
#         # Fallback for older openml versions
#         X, y, categorical_indicator, attribute_names = dataset.get_data(
#             target=dataset.default_target_attribute
#         )
    
#     # 1. Convert Sparse to Dense if necessary
#     # Gisette (ID 41138) often loads as a SparseDataFrame which causes issues with Imputers
#     if hasattr(X, "sparse"):
#         # Pandas < 1.0 style or specific SparseDtype
#         try:
#             X = X.sparse.to_dense()
#         except:
#             pass
    
#     # Ensure X is a DataFrame for type separation
#     if not isinstance(X, pd.DataFrame):
#         X = pd.DataFrame(X, columns=attribute_names)

#     # 2. Split Numeric and Categorical
#     # We handle them separately to apply different imputation strategies
#     num_cols = X.select_dtypes(include=[np.number]).columns
#     cat_cols = X.select_dtypes(exclude=[np.number]).columns
    
#     # Extract values as numpy arrays to avoid "Columns must be same length as key" errors
#     # This happens when Imputer drops a column (all NaNs) and we try to assign back to DF
#     X_num = X[num_cols].values if len(num_cols) > 0 else None
#     X_cat = X[cat_cols].values if len(cat_cols) > 0 else None
    
#     # 3. Impute & Process Numeric Data
#     if X_num is not None:
#         # keep_empty_features=True prevents dropping columns, keeping shapes consistent
#         # If your sklearn version is old (<1.2), this param might not exist. 
#         # The numpy approach below handles shape changes automatically anyway.
#         try:
#             imp_num = SimpleImputer(strategy='mean', keep_empty_features=True)
#         except TypeError:
#             imp_num = SimpleImputer(strategy='mean') # Fallback for older sklearn
            
#         X_num = imp_num.fit_transform(X_num)
        
#     # 4. Impute & Process Categorical Data
#     if X_cat is not None:
#         try:
#             imp_cat = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
#         except TypeError:
#             imp_cat = SimpleImputer(strategy='most_frequent')
            
#         X_cat = imp_cat.fit_transform(X_cat)
        
#         # Ordinal Encode for Tree-based models
#         # We cannot use OneHotEncoder for high-dim data (explodes dimensionality)
#         X_cat_enc = []
#         for i in range(X_cat.shape[1]):
#             le = LabelEncoder()
#             # Force string conversion to handle mixed types
#             col_data = X_cat[:, i].astype(str)
#             X_cat_enc.append(le.fit_transform(col_data))
        
#         if len(X_cat_enc) > 0:
#             X_cat = np.column_stack(X_cat_enc)
#         else:
#             X_cat = None

#     # 5. Concatenate Results
#     if X_num is not None and X_cat is not None:
#         X_final = np.column_stack([X_num, X_cat])
#     elif X_num is not None:
#         X_final = X_num
#     elif X_cat is not None:
#         X_final = X_cat
#     else:
#         raise ValueError("Dataset resulted in 0 features after processing.")
        
#     # 6. Encode Target
#     if y is None:
#         raise ValueError("Target variable is None.")
    
#     # Handle Series or Array
#     y = np.array(y)
#     y_enc = LabelEncoder().fit_transform(y)
    
#     # 7. Final Standardization
#     # Good practice for benchmark stability
#     scaler = StandardScaler()
#     X_final = scaler.fit_transform(X_final)
    
#     print(f"Dataset loaded: {X_final.shape[0]} samples, {X_final.shape[1]} features.")
#     return X_final, y_enc

#########################################################

# # data_loader.py
# import openml
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.impute import SimpleImputer

# def load_openml_dataset(dataset_id):
#     """
#     Fetches dataset from OpenML, handles NaN and encoding.
#     Returns: X (numpy array), y (numpy array), name (str)
#     """
#     print(f"Downloading dataset ID {dataset_id}...")
#     dataset = openml.datasets.get_dataset(dataset_id)
#     X, y, categorical_indicator, attribute_names = dataset.get_data(
#         target=dataset.default_target_attribute
#     )

#     # Convert to DataFrame for easier handling initially
#     df = pd.DataFrame(X, columns=attribute_names)
    
#     # 1. Handle Missing Values (Mean for numeric, Mode for categorical)
#     # Simple strategy for benchmarking
#     num_cols = df.select_dtypes(include=[np.number]).columns
#     cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
#     if len(num_cols) > 0:
#         imp_num = SimpleImputer(strategy='mean')
#         df[num_cols] = imp_num.fit_transform(df[num_cols])
        
#     if len(cat_cols) > 0:
#         imp_cat = SimpleImputer(strategy='most_frequent')
#         df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
#         # Encode Categoricals
#         for col in cat_cols:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))

#     # 2. Encode Target
#     y_enc = LabelEncoder().fit_transform(y)
    
#     # 3. Scale Features (Important for many models, though less so for Trees)
#     # We do it to standardize the benchmark environment
#     X_final = StandardScaler().fit_transform(df)
    
#     print(f"Dataset loaded: {X_final.shape[0]} samples, {X_final.shape[1]} features.")
#     return X_final, y_enc
