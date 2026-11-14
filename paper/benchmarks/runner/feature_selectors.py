# feature_selectors.py
import time
import numpy as np
# from boruta import BorutaPy
from BorutaLocal import BorutaPy
# from greedy_boruta import GreedyBorutaPy 
from GreedyBoruta import GreedyBorutaPy 
from GreedyBorutaParallel import GreedyBorutaPyParallel # Make sure your package is importable

class SelectorWrapper:
    def __init__(self, name, estimator, alpha, random_state=42):
        self.name = name
        self.estimator = estimator
        self.alpha = alpha
        self.selector = None
        self.running_time = 0
        self.n_iterations = 0
        self.random_state = random_state

    def fit(self, X, y, track_history=False):
        start_time = time.time()

        if self.name == "Vanilla":
            self.selector = BorutaPy(
                estimator=self.estimator,
                n_estimators='auto',
                alpha=self.alpha,
                two_step=False,
                max_iter=512,
                early_stopping=True,
                n_iter_no_change=128,
                random_state=self.random_state,
                verbose=0,
                track_history=track_history  # Enable history tracking
            )
        elif self.name == "Greedy":
            self.selector = GreedyBorutaPy(
                estimator=self.estimator,
                n_estimators='auto',
                alpha=self.alpha,
                random_state=self.random_state
            )
        elif self.name == "GreedyParallel":
            self.selector = GreedyBorutaPyParallel(
                estimator=self.estimator,
                n_estimators='auto',
                bonferroni=True, 
                alpha=self.alpha,
                random_state=self.random_state,
                n_jobs=4
            )
            
        self.selector.fit(X, y)
        self.running_time = time.time() - start_time
        
        try:
            self.n_iterations = self.selector._iter
        except:
            self.n_iterations = -1

    def get_confirmed_mask(self):
        """Returns boolean mask of confirmed features"""
        return self.selector.support_

    def get_tentative_mask(self):
        """Returns boolean mask of tentative features (Only relevant for Vanilla)"""
        if hasattr(self.selector, 'support_weak_'):
            return self.selector.support_weak_
        return np.zeros(self.selector.support_.shape, dtype=bool)

    def get_history(self):
        """Returns history if available (Only for Vanilla with track_history=True)"""
        if hasattr(self.selector, 'history_'):
            return self.selector.history_
        return None
    

# #########################################################

# # feature_selectors.py
# import time
# import numpy as np
# # from boruta import BorutaPy
# from BorutaLocal import BorutaPy
# # from greedy_boruta import GreedyBorutaPy 
# from GreedyBoruta import GreedyBorutaPy 
# from GreedyBorutaParallel import GreedyBorutaPyParallel # Make sure your package is importable

# class SelectorWrapper:
#     def __init__(self, name, estimator, alpha):
#         self.name = name
#         self.estimator = estimator
#         self.alpha = alpha
#         self.selector = None
#         self.running_time = 0
#         self.n_iterations = 0

#     def fit(self, X, y):
#         start_time = time.time()
        
#         if self.name == "Vanilla":
#             self.selector = BorutaPy(
#                 estimator=self.estimator,
#                 n_estimators='auto',
#                 alpha=self.alpha,
#                 max_iter=512,          
#                 early_stopping=True,   
#                 n_iter_no_change=64,   
#                 random_state=42,
#                 verbose=0
#             )
#         elif self.name == "Greedy":
#             self.selector = GreedyBorutaPy(
#                 estimator=self.estimator,
#                 n_estimators='auto',
#                 alpha=self.alpha,
#                 random_state=42
#             )
#         elif self.name == "GreedyParallel":
#             self.selector = GreedyBorutaPyParallel(
#                 estimator=self.estimator,
#                 n_estimators='auto',
#                 alpha=self.alpha,
#                 random_state=42,
#                 n_jobs=-1
#             )
            
#         self.selector.fit(X, y)
#         self.running_time = time.time() - start_time
        
#         try:
#             self.n_iterations = self.selector._iter
#         except:

#             self.n_iterations = -1

#     def get_confirmed_mask(self):
#         """Returns boolean mask of confirmed features"""
#         return self.selector.support_

#     def get_tentative_mask(self):
#         """Returns boolean mask of tentative features (Only relevant for Vanilla)"""
#         if hasattr(self.selector, 'support_weak_'):
#             return self.selector.support_weak_
#         return np.zeros(self.selector.support_.shape, dtype=bool)

#########################################################

# # # feature_selectors.py
# # import time
# # import numpy as np
# # from boruta import BorutaPy
# # from greedy_boruta import GreedyBorutaPy # Make sure your package is importable

# # class SelectorWrapper:
# #     def __init__(self, name, estimator, alpha):
# #         self.name = name
# #         self.estimator = estimator
# #         self.alpha = alpha
# #         self.selector = None
# #         self.running_time = 0
# #         self.n_iterations = 0

# #     def fit(self, X, y):
# #         start_time = time.time()
        
# #         if self.name == "Vanilla":
# #             # Standard BorutaPy
# #             self.selector = BorutaPy(
# #                 estimator=self.estimator,
# #                 n_estimators='auto',
# #                 alpha=self.alpha,
# #                 max_iter=100, # Standard default
# #                 random_state=42,
# #                 verbose=0
# #             )
# #         elif self.name == "Greedy":
# #             # Your implementation
# #             self.selector = GreedyBorutaPy(
# #                 estimator=self.estimator,
# #                 n_estimators='auto',
# #                 alpha=self.alpha,
# #                 # Greedy doesn't strictly need max_iter/early_stopping
# #                 # but we pass defaults if your code expects them
# #                 random_state=42
# #             )
            
# #         self.selector.fit(X, y)
# #         self.running_time = time.time() - start_time
        
# #         # Extract iteration count (if available in attributes)
# #         # Depending on implementation, might need to inspect object
# #         try:
# #             self.n_iterations = self.selector.n_iter_
# #         except:
# #             self.n_iterations = -1

# #     def get_support(self):
# #         """Returns boolean mask of selected features"""
# #         return self.selector.support_