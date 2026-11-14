#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Nicolas Vana <nicolas.vana@gmail.com>

Inspired by the code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause

This version extends GreedyBorutaPyParallel to support multiple importance metrics:
- 'feature_importances': Standard feature_importances_ attribute (default)
- 'shap': SHAP values for feature importance
"""

from __future__ import print_function, division
import numpy as np
import scipy as sp
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator, clone
from joblib import Parallel, delayed
import warnings


class GreedyBorutaPyFlexible(BaseEstimator, TransformerMixin):
    """
    Greedy Boruta with Flexible Importance Metrics.

    A faster variant of the Boruta all-relevant feature selection method that
    supports multiple importance metrics including SHAP values.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator compatible with scikit-learn.

    n_estimators : int or 'auto', default=1000
        Number of estimators in the ensemble. If 'auto', calculated automatically.

    perc : int, default=100
        Percentile of shadow feature importance to use as threshold.

    alpha : float, default=0.05
        Significance level for statistical tests.

    two_step : bool, default=True
        Whether to use two-step correction (FDR + Bonferroni).

    importance_metric : str, default='feature_importances'
        Method to calculate feature importance:
        - 'feature_importances': Use estimator.feature_importances_ (default)
        - 'shap': Use SHAP values (requires shap package)

    shap_kwargs : dict, optional
        Additional arguments to pass to SHAP explainer.
        Only used when importance_metric='shap'.
        Common options:
        - 'check_additivity': bool, default=False
        - 'feature_perturbation': str, default='interventional'

    n_jobs : int, default=1
        Number of parallel jobs. -1 uses all processors.

    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    support_ : array of shape [n_features]
        Boolean mask of confirmed features.

    n_features_ : int
        Number of confirmed features.

    ranking_ : array of shape [n_features]
        Feature ranking (1=confirmed, 2=rejected).

    support_weak_ : array of shape [n_features]
        Empty array for compatibility (Greedy has no tentative features).

    max_iter : int
        Number of iterations performed.

    Examples
    --------
    # Standard feature importances
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    >>> selector = GreedyBorutaPyFlexible(rf, n_estimators='auto', n_jobs=-1)
    >>> selector.fit(X, y)

    # SHAP-based importance
    >>> selector_shap = GreedyBorutaPyFlexible(
    ...     rf,
    ...     importance_metric='shap',
    ...     shap_kwargs={'check_additivity': False},
    ...     n_jobs=-1
    ... )
    >>> selector_shap.fit(X, y)
    """

    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 two_step=True, importance_metric='feature_importances',
                 shap_kwargs=None, n_jobs=1, random_state=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.importance_metric = importance_metric
        self.shap_kwargs = shap_kwargs if shap_kwargs is not None else {}
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.__version__ = '0.5-flexible'
        self._is_lightgbm = 'lightgbm' in str(type(self.estimator))

        # Validate importance metric
        valid_metrics = ['feature_importances', 'shap']
        if self.importance_metric not in valid_metrics:
            raise ValueError(
                f"importance_metric must be one of {valid_metrics}, "
                f"got '{self.importance_metric}'"
            )

        # Check SHAP availability if needed
        if self.importance_metric == 'shap':
            try:
                import shap
                self._shap_available = True
            except ImportError:
                raise ImportError(
                    "SHAP importance metric requires the 'shap' package. "
                    "Install it with: pip install shap"
                )

    def fit(self, X, y):
        return self._fit(X, y)

    def transform(self, X, return_df=False):
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')

        if return_df:
            X = X.iloc[:, self.support_]
        else:
            X = X[:, self.support_]
        return X

    def fit_transform(self, X, y, return_df=False):
        self._fit(X, y)
        return self.transform(X, return_df)

    def _validate_pandas_input(self, arg):
        try:
            return arg.values
        except AttributeError:
            raise ValueError(
                "input needs to be a numpy array or pandas data frame."
            )

    @staticmethod
    def _get_importance_standard(estimator):
        """Extract standard feature_importances_ from estimator."""
        try:
            imp = estimator.feature_importances_
        except Exception:
            raise ValueError('Estimator must have feature_importances_ attribute.')
        return imp

    @staticmethod
    def _get_importance_shap(estimator, X, y, shap_kwargs):
        """Calculate SHAP-based feature importance."""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP package required for SHAP importance.")

        # Determine which SHAP explainer to use based on estimator type
        estimator_type = type(estimator).__name__.lower()

        try:
            # For tree-based models, use TreeExplainer (faster)
            if any(tree_type in estimator_type for tree_type in
                   ['forest', 'tree', 'xgb', 'lgbm', 'catboost', 'gbm']):

                # TreeExplainer doesn't need background data
                explainer = shap.TreeExplainer(estimator, **shap_kwargs)
                shap_values = explainer.shap_values(X)

            else:
                # For other models, use KernelExplainer or LinearExplainer
                # Use a sample of data as background for KernelExplainer (faster)
                if len(X) > 100:
                    # Sample background data to speed up computation
                    rng = np.random.RandomState(42)
                    bg_indices = rng.choice(len(X), size=min(100, len(X)), replace=False)
                    background = X[bg_indices]
                else:
                    background = X

                # Predict function for KernelExplainer
                if hasattr(estimator, 'predict_proba'):
                    predict_fn = estimator.predict_proba
                else:
                    predict_fn = estimator.predict

                explainer = shap.KernelExplainer(predict_fn, background, **shap_kwargs)
                shap_values = explainer.shap_values(X)

            # Handle multi-class classification (shap_values is a list)
            if isinstance(shap_values, list):
                # For multi-class, average absolute SHAP values across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

            # Calculate mean absolute SHAP value for each feature
            # Shape: (n_features,)
            importance = np.mean(np.abs(shap_values), axis=0)

            return importance

        except Exception as e:
            raise ValueError(f"SHAP importance calculation failed: {str(e)}")

    @staticmethod
    def _parallel_fit_iteration(estimator, X, y, random_state, perc, is_lightgbm,
                                importance_metric, shap_kwargs):
        """
        Static worker to run one iteration of shadow feature creation and model fitting.
        Returns a boolean array indicating which features were hits.

        Parameters
        ----------
        importance_metric : str
            'feature_importances' or 'shap'
        shap_kwargs : dict
            Arguments for SHAP explainer
        """
        # Setup random state for this specific job
        rng = check_random_state(random_state)

        # Handle LightGBM specific randomness
        if is_lightgbm:
            estimator.set_params(random_state=rng.randint(0, 10000))
        else:
            estimator.set_params(random_state=rng)

        n_samples, n_features = X.shape

        # 1. Create Shadow Features
        x_sha = np.copy(X)

        # Shuffle each column individually to destroy relationship with Y
        for i in range(n_features):
            rng.shuffle(x_sha[:, i])

        # Ensure we have enough shadows (legacy Boruta behavior: at least 5)
        while x_sha.shape[1] < 5:
            x_sha = np.hstack((x_sha, x_sha))

        # 2. Fit Estimator on X + Shadows
        X_combined = np.hstack((X, x_sha))

        try:
            estimator.fit(X_combined, y)
        except Exception as e:
            raise ValueError('Estimator fit failed.\n' + str(e))

        # 3. Extract Importances (using specified metric)
        if importance_metric == 'feature_importances':
            imp = GreedyBorutaPyFlexible._get_importance_standard(estimator)
        elif importance_metric == 'shap':
            imp = GreedyBorutaPyFlexible._get_importance_shap(
                estimator, X_combined, y, shap_kwargs
            )
        else:
            raise ValueError(f"Unknown importance metric: {importance_metric}")

        # Separate real and shadow importances
        imp_real = imp[:n_features]
        imp_sha = imp[n_features:]

        # 4. Determine Hits
        # Calculate shadow threshold
        imp_sha_max = np.percentile(imp_sha, perc)

        # Handle NaNs in importance (treat as 0/no-hit)
        cur_imp_no_nan = np.nan_to_num(imp_real)

        # A hit is when feature importance > max shadow importance
        hits = (cur_imp_no_nan > imp_sha_max)

        return hits

    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X)
        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        self.random_state = check_random_state(self.random_state)

        n_sample, n_feat = X.shape

        # --- 1. Determine Convergence Criteria ---
        self.max_iter = self._compute_iterations_to_convergence(
            self.alpha, n_feat, self.two_step
        )

        # --- 2. Setup Estimator ---
        # Calculate n_estimators once based on full feature set
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)
        else:
            n_tree = self._get_tree_num(n_feat)
            self.estimator.set_params(n_estimators=n_tree)

        if self.verbose > 0:
            print(f"GreedyBoruta (Flexible): Running {self.max_iter} iterations.")
            print(f"  Importance metric: {self.importance_metric}")
            print(f"  n_jobs: {self.n_jobs}")

        # --- 3. Generate Seeds ---
        seeds = self.random_state.randint(0, 2**31 - 1, size=self.max_iter)

        # --- 4. Parallel Execution ---
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._parallel_fit_iteration)(
                clone(self.estimator),
                X,
                y,
                s,
                self.perc,
                self._is_lightgbm,
                self.importance_metric,
                self.shap_kwargs
            ) for s in seeds
        )

        # --- 5. Decision Logic ---
        all_hits = np.array(results)
        total_hits = np.sum(all_hits, axis=0)

        self.support_ = (total_hits > 0)
        self.n_features_ = np.sum(self.support_)
        self.ranking_ = np.where(self.support_, 1, 2)

        # For compatibility with standard Boruta inspections
        self.support_weak_ = np.zeros(n_feat, dtype=bool)

        if self.verbose > 0:
            print(f"Finished. Confirmed: {self.n_features_}, Rejected: {n_feat - self.n_features_}")

        return self

    def _compute_iterations_to_convergence(self, alpha, n_features, two_step):
        """
        Calculates the number of iterations K required such that the probability
        of a random feature having 0 hits is less than alpha (or alpha/n_features).

        Solves: 0.5^K < Threshold
        => K > log2(1/Threshold)
        """
        if two_step:
            threshold = alpha
        else:
            # Bonferroni correction for number of features
            threshold = alpha / float(n_features)

        # Calculate K
        try:
            k_float = np.log2(1.0 / threshold)
        except ZeroDivisionError:
            raise ValueError("Alpha must be greater than 0.")

        return int(np.ceil(k_float))

    def _get_tree_num(self, n_feat):
        depth = None
        try:
            depth = self.estimator.get_params()['max_depth']
        except KeyError:
            warnings.warn(
                "The estimator does not have a max_depth property, as a result "
                " the number of trees to use cannot be estimated automatically."
            )
        if depth == None:
            depth = 10
        f_repr = 100
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _check_params(self, X, y):
        check_X_y(X, y)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')
