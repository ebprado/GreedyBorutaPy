#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Nicolas Vana <nicolas.vana@gmail.com>

Inspired by the code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""

from __future__ import print_function, division
import numpy as np
import scipy as sp
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator, clone
from joblib import Parallel, delayed
import warnings
from numbers import Integral, Real


class GreedyBorutaPyParallel(BaseEstimator, TransformerMixin):
    """
    Greedy Boruta: A faster variant of the Boruta all-relevant feature selection method.
    
    ... [Docstring abbreviated for brevity] ...
    
    Parameters
    ----------
    n_jobs : int, default=1
        The number of jobs to run in parallel. -1 means using all processors.
    n_shadows : None, int or float, default=None
        None uses all p features. An integer specifies 1 <= r <= p shadows;
        a float in (0, 1] uses ceil(p * n_shadows) shadows. At each iteration,
        r distinct original columns are sampled uniformly and independently
        permuted. The iteration budget still depends on p, not r.
        The K/(r+1) bound assumes shadow exchangeability and perc=100.
        
    ... [Other parameters same as before] ...
    """
    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 bonferroni=True, n_jobs=1, random_state=None, verbose=0,
                 n_shadows=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.bonferroni = bonferroni
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.n_shadows = n_shadows
        self.__version__ = '0.4-parallel'
        self._is_lightgbm = 'lightgbm' in str(type(self.estimator))

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
    def _parallel_fit_iteration(estimator, X, y, random_state, perc, is_lightgbm,
                                n_shadows=None):
        """
        Static worker to run one iteration of shadow feature creation and model fitting.
        Returns a boolean array indicating which features were hits.
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
        # We copy X to create shadows
        r = n_features if n_shadows is None else n_shadows
        # Avoid consuming extra randomness in the full-shadow case.
        columns = np.arange(n_features) if r == n_features else rng.choice(n_features, r, replace=False)
        x_sha = np.copy(X[:, columns])
        
        # # Shuffle each column individually to destroy relationship with Y
        # for i in range(n_features):
        #     rng.shuffle(x_sha[:, i])

        # Shuffle each column in-place, matching vanilla Boruta's approach
        def _shuffle_column(col):
            rng.shuffle(col)
            return col
        
        x_sha = np.apply_along_axis(_shuffle_column, 0, x_sha)

        # idx = np.argsort(rng.random((n_samples, n_features)), axis=0)
        # x_sha = x_sha[idx, np.arange(n_features)]
            
        # Use exactly r shadows, including when r < 5.
            
        # 2. Fit Estimator on X + Shadows
        X_combined = np.hstack((X, x_sha))
        
        try:
            estimator.fit(X_combined, y)
        except Exception as e:
            raise ValueError('Estimator fit failed.\n' + str(e))
            
        # 3. Extract Importances
        try:
            imp = estimator.feature_importances_
        except Exception:
            raise ValueError('Estimator must have feature_importances_ attribute.')
            
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

        rng = check_random_state(self.random_state)
        
        n_sample, n_feat = X.shape
        self.n_features_in_ = n_feat
        self.n_shadows_ = self._resolve_n_shadows(n_feat)

        # --- 1. Determine Convergence Criteria ---
        # Rejection uses 0.5^K <= alpha (or alpha/n_features).
        self.max_iter = self._compute_iterations_to_convergence(
            self.alpha, n_feat)
        
        # --- 2. Setup Estimator ---
        # Calculate n_estimators once based on full feature set
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)
        else:
            n_tree = self._get_tree_num(n_feat)
            self.estimator.set_params(n_estimators=n_tree)

        if self.verbose > 0:
            print(f"GreedyBoruta: Running {self.max_iter} iterations. n_jobs={self.n_jobs}")

        # --- 3. Generate Seeds (The Key to Determinism) ---
        # We generate seeds upfront. This ensures Iteration 2 uses the same seed
        # whether it runs immediately (Parallel) or after Iteration 1 (Serial).
        seeds = rng.randint(0, 2**31 - 1, size=self.max_iter)

        # --- 4. Execution ---
        # joblib handles the loop. If n_jobs=1, it runs sequentially.
        # If n_jobs=-1, it runs in parallel.
        # The results are identical because 'seeds' is identical.
        # results = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self._parallel_fit_iteration)(
        #         clone(self.estimator), 
        #         X, 
        #         y, 
        #         s, 
        #         self.perc, 
        #         self._is_lightgbm
        #     ) for s in seeds
        # )
        if self.n_jobs == 1:
            results = [
                self._parallel_fit_iteration(
                    clone(self.estimator), X, y, s, self.perc, self._is_lightgbm, self.n_shadows_
                ) for s in seeds
            ]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._parallel_fit_iteration)(
                    clone(self.estimator), X, y, s, self.perc, self._is_lightgbm, self.n_shadows_
                ) for s in seeds
            )
        
        # --- 5. Decision Logic ---
        all_hits = np.array(results)
        total_hits = np.sum(all_hits, axis=0)
        self.hit_counts_ = total_hits
        self.n_iter_ = self.max_iter

        self.support_ = (total_hits > 0)
        self.n_features_ = np.sum(self.support_)
        self.ranking_ = np.where(self.support_, 1, 2)
        
        # For compatibility with standard Boruta inspections (though empty/unused)
        self.support_weak_ = np.zeros(n_feat, dtype=bool)

        if self.verbose > 0:
            print(f"Finished. Confirmed: {self.n_features_}, Rejected: {n_feat - self.n_features_}")
            
        return self

    # def _compute_iterations_to_convergence(self, alpha, n_features):
    #     """
    #     Calculates the number of iterations K required such that the probability
    #     of a random feature having 0 hits is less than alpha (or alpha/n_features).
        
    #     Solves: 0.5^K < Threshold
    #     => K > log2(1/Threshold)
    #     """
    #     if not self.bonferroni:
    #         # FDR with identical p-values (0 hits) implies we check against alpha.
    #         # We do NOT divide by K (temporal correction) because we don't peek.
    #         threshold = alpha / float(n_features)
    #     else:
    #         # Bonferroni correction for number of features
    #         threshold = alpha

    #     # Calculate K
    #     # 2^K > 1/threshold  =>  K > log2(1/threshold)
    #     try:
    #         k_float = np.log2(1.0 / threshold)
    #     except ZeroDivisionError:
    #         raise ValueError("Alpha must be greater than 0.")
            
    #     # We need the integer strictly greater than k_float, so ceil is appropriate
    #     return int(np.ceil(k_float))

    def _compute_iterations_to_convergence(self, alpha, n_features):
        """
        Calculates the number of iterations K required for rejection of 0-hit features.
        
        In Greedy Boruta, all tentative features have exactly 0 hits, giving them
        identical p-values of 0.5^K. To reject at significance level alpha (with
        optional Bonferroni correction for multiple features), we need:
        
            0.5^K <= threshold
            => K >= log2(1/threshold)
            => K = ceil(log2(1/threshold))
        
        Parameters
        ----------
        alpha : float
            Significance level for the rejection test.
        n_features : int
            Number of features in the dataset, used for Bonferroni correction.
        
        Returns
        -------
        int
            Minimum number of iterations K guaranteeing convergence.
        
        Notes
        -----
        With bonferroni=True (default): threshold = alpha/n_features
            Matches vanilla R Boruta with mcAdj=TRUE (the R default).
            Applies the feature-wise Bonferroni threshold to rejection;
            it does not control FDR among single-hit confirmations.
        
        With bonferroni=False: threshold = alpha
            Matches vanilla R Boruta with mcAdj=FALSE.
            No correction for multiple comparisons.
        """
        if self.bonferroni:
            # Bonferroni correction: divide alpha by number of features
            # This matches vanilla R Boruta's default behavior (mcAdj=TRUE)
            threshold = alpha / float(n_features)
        else:
            # No correction for multiple comparisons
            # Matches vanilla R Boruta with mcAdj=FALSE
            threshold = alpha

        # Solve 0.5^K <= threshold for K
        # => K >= log2(1/threshold)
        # => K = ceil(log2(1/threshold))
        if threshold <= 0:
            raise ValueError("Alpha must be greater than 0.")
        
        k_float = np.log2(1.0 / threshold)
        return int(np.ceil(k_float))

    def _resolve_n_shadows(self, n_features):
        value = self.n_shadows
        if value is None:
            return n_features
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('n_shadows must be None, an integer count, or a fraction in (0, 1].')
        if isinstance(value, Integral) and 1 <= value <= n_features:
            return int(value)
        if isinstance(value, Real) and not isinstance(value, Integral) and 0 < value <= 1:
            return int(np.ceil(n_features * value))
        raise ValueError('n_shadows must be an integer in [1, p] or a fraction in (0, 1].')

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
