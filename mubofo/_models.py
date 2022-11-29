from __future__ import annotations

import logging

from typing import Any, Optional

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
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
        val_size: Optional[int | float] = None,
        max_patience: Optional[int] = None,
        threshold: float = 0,
        random_state: Optional[int | np.random.RandomState] = None,
        verbose: bool = False
    ) -> None:
        """
        Initialize a BoostedForestRegressor and set parameters.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest. If val_size is not None, n_estimators
            is the maximum number of trees that might be trained.
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
        val_size : int or float or None
            If None, no early stopping occurs. If an int or a float, will be
            passed as the test_size argument to train_test_split to use a
            subset of the training data as a validation set and stop training
            early when the validation error has not decreased by at least
            threshold in max_patience iterations.
        max_patience : int or None
            Number of boosting iterations to wait for the validation score to
            decrease by threshold before stopping early. If max_patience is None
            but val_size is not None, max_patience will be set to one-tenth the
            total number of estimators to be trained.
        threshold : float
            Decrement by which the validation error must fall in max_patience
            iterations to prevent early stopping. The validation error is the
            RMS error of the model predictions, averaged over all outputs if
            there are multiple.
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

        self.val_size = val_size
        self.max_patience = max_patience
        self.threshold = threshold

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

        validate = self.val_size is not None
        if validate:
            X, X_va, Y, Y_va = train_test_split(
                X, Y,
                test_size=self.val_size,
                random_state=random_state
            )

        n_samples, n_features = X.shape
        max_samples = self.max_samples

        if max_samples is None:
            max_samples = n_samples
        elif isinstance(max_samples, float):
            max_samples = max(round(max_samples * n_samples), 1)

        current = np.zeros(Y.shape)
        estimators: list[DecisionTreeRegressor] = []

        if validate:
            current_va = np.zeros(Y_va.shape)
            best_error = np.sqrt((Y_va ** 2).mean())

            patience, max_patience = 0, self.max_patience    
            if max_patience is None:
                max_patience = max(round(0.1 * self.n_estimators), 1)

        for i in range(1, self.n_estimators + 1):
            weights = ((Y - current) ** 2).reshape(n_samples, -1).mean(axis=1)
            weights = weights / weights.sum()
            idx = random_state.choice(n_samples, size=max_samples, p=weights)

            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=random_state
            ).fit(X[idx], Y[idx] - current[idx])

            estimators.append(estimator)
            current += self.learning_rate * estimator.predict(X)
            message = f'fit estimator {i}'

            if validate:
                current_va += self.learning_rate * estimator.predict(X_va)
                error = np.sqrt(((Y_va - current_va) ** 2).mean())
                message += f' -- validation error is {error:.3f}'

                if error < best_error - self.threshold:
                    best_error = error
                    patience = 0
                else:
                    patience = patience + 1

                if patience == max_patience:
                    estimators = estimators[:-max_patience]
                    message += '\nTerminating early due to lack of improvement'

            if self.verbose:
                logging.info(message)

            if len(estimators) < i:
                break

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
        X = check_array(X, dtype=np.float32)        
        output = sum(e.predict(X, check_input=False) for e in self.estimators_)

        return self.learning_rate * output

    def _more_tags(self) -> dict[str, Any]:
        return {'multioutput' : True}