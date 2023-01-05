from __future__ import annotations

import logging

from typing import Any, Optional

import numpy as np

from numpy.random import RandomState
from sklearn.ensemble._forest import ForestRegressor, _get_n_samples_bootstrap
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y

from ._trees import MultioutputDecisionTree

class MultioutputForestMixin:
    """Mixin with weighted feature importances and correct tags."""

    estimators_: list[MultioutputDecisionTree]
    n_features_in_: int
    n_outputs_: int

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Calculate feature importances weighted by tree absolute means.
        
        Returns
        -------
        importances : Per-output feature importances for the whole forest. If
            `self.n_outputs_ > 1`, the first axis indexes over input features
            and the second over outputs.

        """

        importances = np.zeros((self.n_features_in_, self.n_outputs_))
        weights = np.array([e.abs_means_ for e in self.estimators_])

        for i, estimator in enumerate(self.estimators_):
            importances += weights[i] * estimator.feature_importances_

        sums = weights.sum(axis=0)
        importances[:, sums != 0] /= sums[sums != 0]

        return importances

    def _more_tags(self) -> dict[str, Any]:
        """Specify that the forest supports multioutput problems."""

        return {'multioutput' : True}

class MultioutputBoostedForest(MultioutputForestMixin, ForestRegressor):
    """Boosted forest regressor with native multioutput support."""

    def __init__(
        self,
        n_estimators: int=100,
        learning_rate: float=0.1,
        max_depth: Optional[int]=None,
        max_samples: Optional[int | float]=None,
        max_features: Optional[int | float]=None,
        val_size: Optional[int | float]=None,
        max_patience: Optional[int]=None,
        threshold: float=0,
        random_state: Optional[int | RandomState]=None,
        logging: bool=False
    ) -> None:
        """
        Initialize a `MultioutputBoostedForest` and set parameters.
        
        Parameters
        ----------
        learning_rate : Weight multiplying the output of each tree.
        val_size : If `None`, no early stopping occurs. If an int or a float,
            will be passed as the `test_size` argument to `train_test_split` to
            use a subset of the training data as a validation set and stop
            training early when the validation error has not decreased by at
            least `threshold` in `max_patience` iterations.
        max_patience : Number of boosting iterations to wait for the validation
            score to decrease by `threshold` before stopping early. If
            `max_patience` is `None` but `val_size` is not `None`,
            `max_patience` will be set to one-tenth the total number of
            estimators to be trained.
        threshold : Decrement by which the validation error must fall in
            `max_patience` iterations to prevent early stopping. The validation
            error is the RMS error of the model predictions, averaged over all
            outputs if there are multiple.
        logging : Whether to log progress reports during fitting.

        Other parameters are as in `RandomForestRegressor`.

        """

        super().__init__(
            base_estimator=MultioutputDecisionTree(),
            n_estimators=n_estimators,
            estimator_params=(
                'max_depth',
                'max_features',
                'random_state'
            ),
            max_samples=max_samples,
            random_state=random_state
        )

        self.learning_rate = learning_rate
        self.logging = logging

        self.max_depth = max_depth
        self.max_features = max_features

        self.val_size = val_size
        self.max_patience = max_patience
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> MultioutputBoostedForest:
        """
        Fit a boosted forest on the training set (X, y).

        Parameters
        ----------
        X : Training inputs.
        y : Training targets.

        Returns
        -------
        MultioutputBoostedForest
            The trained regressor.
        
        """

        self._validate_estimator()
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if y.ndim > 1 and y.shape[1] == 1:
            y = y.flatten()

        random_state = self.random_state
        if random_state is None or isinstance(random_state, int):
            random_state = RandomState(random_state)

        validate = self.val_size is not None
        if validate:
            X, X_va, y, y_va = train_test_split(
                X, y,
                test_size=self.val_size,
                random_state=random_state
            )

        n_samples, n_features = X.shape
        max_samples = _get_n_samples_bootstrap(n_samples, self.max_samples)

        current = np.zeros(y.shape)
        estimators: list[MultioutputDecisionTree] = []

        if validate:
            current_va = np.zeros(y_va.shape)
            best_error = np.sqrt((y_va ** 2).mean())

            patience, max_patience = 0, self.max_patience
            if max_patience is None:
                max_patience = max(round(0.1 * self.n_estimators), 1)

        for i in range(1, self.n_estimators + 1):
            errors = ((y - current) ** 2).reshape(n_samples, -1).mean(axis=1)
            errors = errors / errors.sum()

            idx = random_state.choice(n_samples, size=max_samples, p=errors)
            sample_counts = np.bincount(idx, minlength=n_samples)
            sample_weight = sample_counts * np.ones(n_samples)

            estimator = self._make_estimator(False, random_state)
            estimator.fit(X, y - current, sample_weight)
            current += self.learning_rate * estimator.predict(X)

            estimators.append(estimator)
            message = f'fit estimator {i}'

            if validate:
                current_va += self.learning_rate * estimator.predict(X_va)
                error = np.sqrt(((y_va - current_va) ** 2).mean())
                message += f' -- validation error is {error:.3f}'

                patience = patience + 1
                if error < best_error - self.threshold:
                    best_error = error
                    patience = 0

                if patience == max_patience:
                    estimators = estimators[:-max_patience]
                    message += '\nterminating early due to lack of improvement'

            if self.logging:
                logging.info(message)

            if len(estimators) < i:
                break

        self.estimators_ = estimators
        self.n_features_in_ = n_features
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make a prediction with the fitted regressor.

        Parameters
        ----------
        X : Input features to make predictions for.

        Returns
        -------
        output : Regressor predictions.

        """
        
        output = super().predict(X)
        output *= self.learning_rate * len(self.estimators_)

        return output

class MultioutputRandomForest(MultioutputForestMixin, ForestRegressor):
    """Random forest regressor with multioutput feature importances."""

    def __init__(
        self,
        n_estimators: int=100,
        criterion: str='squared_error',
        max_depth: Optional[int]=None,
        max_samples: Optional[int | float]=None,
        max_features: Optional[int | float]=None,
        random_state: Optional[int | np.random.RandomState]=None
    ) -> None:
        """Initialize a random forest, as in `RandomForestRegressor`."""

        super().__init__(
            base_estimator=MultioutputDecisionTree(),
            n_estimators=n_estimators,
            estimator_params=(
                'criterion',
                'max_depth',
                'max_features',
                'random_state'
            ),
            max_samples=max_samples,
            random_state=random_state,
            bootstrap=True
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X: np.ndarray, y: np.ndarray, **_) -> MultioutputRandomForest:
        """
        Fit the random forest, ignoring keyword arguments.
        
        `X` and `y` are as in `RandomForestRegressor`.
        
        """

        return super().fit(X, y)