"""Regression checks for training-fold-only feature selection."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import evaluator


class RecordingClassifier(DecisionTreeClassifier):
    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        return super().fit(X, y)


class RecordingRegressor(DecisionTreeRegressor):
    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        return super().fit(X, y)


class RecordingWrapper:
    def __init__(self, name="Vanilla", estimator=None, alpha=0.05, random_state=0):
        self.name = name
        self.estimator = estimator if estimator is not None else DecisionTreeClassifier()
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y, track_history=False):
        self.train_ids = X[:, 0].copy()
        self.train_y = y.copy()
        self.track_history = track_history

    def get_confirmed_mask(self):
        return np.array([True, False, False])

    def get_tentative_mask(self):
        return np.array([False, True, False])

    def get_history(self):
        return [{'iteration': 1, 'confirmed': {1}},
                {'iteration': 2, 'confirmed': {2}}]


@pytest.mark.parametrize("task_type", ["class", "reg"])
def test_selection_sees_only_training_rows_and_labels(task_type):
    X = np.arange(180).reshape(60, 3)
    y = np.arange(60) % 2 if task_type == "class" else np.arange(60) / 7
    folds = evaluator.fit_cv_selectors(X, y, RecordingWrapper(), task_type, 7)
    assert len(folds) == 5
    assert len({id(selector) for _, _, selector in folds}) == 5
    for train, test, selector in folds:
        np.testing.assert_array_equal(selector.train_ids, X[train, 0])
        np.testing.assert_array_equal(selector.train_y, y[train])
        assert set(selector.train_ids).isdisjoint(X[test, 0])
        assert selector.track_history


@pytest.mark.parametrize("task_type", ["class", "reg"])
def test_scores_use_fold_masks_for_final_weak_and_checkpoints(monkeypatch, task_type):
    X = np.arange(180).reshape(60, 3)
    y = np.arange(60) % 2 if task_type == "class" else np.arange(60) / 7
    folds = evaluator.fit_cv_selectors(X, y, RecordingWrapper(), task_type, 7)
    monkeypatch.setattr(evaluator, 'LGBMClassifier', lambda **kw: RecordingClassifier())
    monkeypatch.setattr(evaluator, 'LGBMRegressor', lambda **kw: RecordingRegressor())
    for kwargs, columns in [({}, [0]), ({'include_weak': True}, [0, 1]),
                            ({'iteration': 1}, [1]), ({'iteration': 100}, [2])]:
        seen = []

        def score(model, held_out_X, held_out_y):
            train, test, _ = folds[len(seen)]
            np.testing.assert_array_equal(model.fit_X, X[train][:, columns])
            np.testing.assert_array_equal(model.fit_y, y[train])
            np.testing.assert_array_equal(held_out_X, X[test][:, columns])
            np.testing.assert_array_equal(held_out_y, y[test])
            assert model.n_features_in_ == len(columns)
            seen.append(test)
            return 0.5

        monkeypatch.setattr(evaluator, 'get_scorer', lambda name: score)
        assert evaluator.evaluate_downstream(X, y, folds, task_type, 7, **kwargs) == (0.5, 0.0)
        assert len(seen) == 5
    assert evaluator.evaluate_downstream(X, y, folds, task_type, 7, iteration=0) == (0.0, 0.0)
