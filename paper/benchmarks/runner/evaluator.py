import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
from sklearn.metrics import get_scorer

def fit_cv_selectors(X, y, wrapper, task_type="class", random_state=42):
    """Fit a fresh selector on each training fold; reuse fits across score variants."""
    if task_type == "class":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    elif task_type == "reg":
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        raise ValueError("Unknown task_type")
    folds = []
    for train, test in cv.split(X, y):
        selector = type(wrapper)(wrapper.name, clone(wrapper.estimator), wrapper.alpha,
                                 random_state=wrapper.random_state)
        selector.fit(X[train], y[train], track_history=(wrapper.name == "Vanilla"))
        folds.append((train, test, selector))
    return folds


def evaluate_downstream(X, y, folds, task_type="class", random_state=42,
                        include_weak=False, iteration=None):
    """
    Evaluate LightGBM with training-fold-only feature masks from fit_cv_selectors.
    An iteration selects the corresponding training-fold checkpoint, never a
    mask fitted on the full dataset. After early stopping, reuse the last state.
    - Classification: F1 Weighted (Stratified CV)
    - Regression: R2 Score (KFold CV)
    """
    if task_type == "class":
        clf = LGBMClassifier(
            n_estimators=256, max_depth=5, learning_rate=0.1,
            n_jobs=-1, random_state=random_state, verbose=-1
        )
        scoring = 'f1_weighted'
        
    elif task_type == "reg":
        clf = LGBMRegressor(
            n_estimators=256, max_depth=5, learning_rate=0.1,
            n_jobs=-1, random_state=random_state, verbose=-1
        )
        scoring = 'r2' # Coefficient of determination
        
    else:
        raise ValueError("Unknown task_type")
    
    scores = []
    scorer = get_scorer(scoring)
    for train, test, selector in folds:
        if iteration is None:
            mask = selector.get_confirmed_mask().copy()
            if include_weak:
                mask |= selector.get_tentative_mask()
        else:
            history = selector.get_history()
            if history is None:
                raise ValueError("Checkpoint evaluation requires training-fold history")
            states = [state for state in history if state['iteration'] <= iteration]
            mask = np.zeros(X.shape[1], dtype=bool)
            if states:
                mask[list(states[-1]['confirmed'])] = True
        if not np.any(mask):
            scores.append(0.0)  # Preserve the benchmark's empty-selection convention.
            continue
        model = clone(clf)
        model.fit(X[train][:, mask], y[train])
        scores.append(scorer(model, X[test][:, mask], y[test]))
    return np.mean(scores), np.std(scores)
    
#########################################################

# # evaluator.py
# import numpy as np
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold

# def evaluate_downstream(X, y, mask, random_state=42):
#     """
#     Evaluates the selected feature subset using 5-Fold Cross-Validation
#     with a LightGBM classifier.
    
#     Returns:
#         mean_f1 (float): Average weighted F1 score across 5 folds.
#         std_f1 (float): Standard deviation of F1 score.
#     """
#     # 1. Handle Empty Selection
#     # If the algorithm rejects EVERYTHING, we return a baseline score (random guess)
#     if np.sum(mask) == 0:
#         return 0.0, 0.0 
    
#     # 2. Filter Features
#     X_filtered = X[:, mask]
    
#     # 3. Setup LightGBM
#     # We use a fixed configuration to ensure fair comparison
#     clf = LGBMClassifier(
#         n_estimators=256,
#         max_depth=5,
#         learning_rate=0.1,
#         n_jobs=2, # limit threads per fold to avoid congestion
#         random_state=random_state,
#         verbose=-1
#     )
    
#     # 4. 5-Fold Stratified CV
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
#     try:
#         # We focus on F1 Weighted as it balances precision/recall well for class imbalance
#         scores = cross_val_score(clf, X_filtered, y, cv=cv, scoring='f1_weighted')
#         return scores.mean(), scores.std()
#     except Exception as e:
#         print(f"    [Eval Error] {e}")
#         return 0.0, 0.0

# # # evaluator.py
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, f1_score
# # from sklearn.model_selection import train_test_split
# # import numpy as np

# # def evaluate_downstream(X, y, mask, random_state=42):
# #     """
# #     Trains a standard classifier on the SELECTED features only
# #     to see if predictive power was retained.
# #     """
# #     if np.sum(mask) == 0:
# #         return 0.5, 0.0 # Return baseline if no features selected
    
# #     # Filter features
# #     X_filtered = X[:, mask]
    
# #     # Split
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X_filtered, y, test_size=0.3, random_state=random_state
# #     )
    
# #     # Train a simple Validator Model
# #     # Note: We use a fresh RF to evaluate power, completely separate from the selector
# #     clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
# #     clf.fit(X_train, y_train)
# #     y_pred = clf.predict(X_test)
    
# #     acc = accuracy_score(y_test, y_pred)
# #     f1 = f1_score(y_test, y_pred, average='weighted')
    
# #     return acc, f1
