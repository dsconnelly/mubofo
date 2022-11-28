from __future__ import annotations

import logging

from typing import Any, Optional

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class BoostedForestRegressor(BaseEstimator, RegressorMixin):
    """Boosted forest regressor with native multioutput support."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        max_samples: Optional[int | float] = None,
        max_features: Optional[int | float] = None,
        random_state: Optional[int | np.random.RandomState] = None,
        verbose: bool = False
    ) -> None:
        """
        Initialize a BoostedForestRegressor and set parameters.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.
        learning_rate : float
            Weight multiplying the output of each tree.
        max_depth : int or None 
            Maximum depth of each tree. See the documentation for
            DecisionTreeRegressor for more details.
        max_samples : int or float or None
            Size of each bootstrapped subsample of the dataset. If None,
            then each subsample will have n_samples rows. If an int, each
            subsample will have max_samples rows. If a float between 0 and 1,
            each subsample will have int(max_samples * n_samples) rows.
        max_features : int or float or None
            Number of features to consider when looking for the best split. See
            the documentation for DecisionTreeRegressor for more details.
        random_state : int or np.random.RandomState or None
            Random state to use for subsampling and to pass to each tree. If
            None, a RandomState is created with an unpredictable seed from the
            system. If an int, one is created with random_state as its seed.
        verbose : bool
            Whether to print progress reports during fitting.

        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.max_depth = max_depth
        self.max_samples = max_samples
        self.max_features = max_features

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, Y: np.ndarray) -> BoostedForestRegressor:
        """
        Fit a boosted forest on the training set (X, Y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training inputs.
        Y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The training targets.

        Returns
        -------
        BoostedForestRegressor
            The fitted regressor.
        
        """

        X, Y = check_X_y(X, Y, multi_output=True, y_numeric=True)
        if Y.ndim > 1 and Y.shape[1] == 1:
            Y = Y.flatten()

        random_state = self.random_state
        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        n_samples, n_features = X.shape
        max_samples = self.max_samples

        if max_samples is None:
            max_samples = n_samples
        elif isinstance(max_samples, float):
            max_samples = max(round(max_samples * n_samples), 1)

        current = np.zeros(Y.shape)
        estimators: list[DecisionTreeRegressor] = []

        for i in range(1, self.n_estimators + 1):
            errors = ((Y - current) ** 2).reshape(n_samples, -1).mean(axis=1)
            weights = errors / errors.sum()
            idx = random_state.choice(n_samples, size=max_samples, p=weights)

            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=random_state
            ).fit(X[idx], Y[idx] - current[idx])

            estimators.append(estimator)
            current += self.learning_rate * estimator.predict(X)

            if self.verbose:
                logging.info(f'Fit estimator {i}.')

        self.estimators_ = estimators
        self.n_features_in_ = n_features

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make a prediction with the fitted regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The inputs to make predictions for.

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_features)
            The regressor's predictions.

        """

        check_is_fitted(self)
        X = check_array(X)

        output = sum(estimator.predict(X) for estimator in self.estimators_)
        output = self.learning_rate * output

        return output

    def _more_tags(self) -> dict[str, Any]:
        return {'multioutput' : True}