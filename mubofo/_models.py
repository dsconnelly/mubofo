import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._utils import get_error, get_n_bootstrap

class BoostedForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.2,
        max_samples=None,
        max_depth=5,
        max_features=None,
        val_size=None,
        threshold=0.01,
        patience=10,
        random_state=None,
        verbose=False
    ):
        """
        Initializes a BoostedForestRegressor and sets some attributes.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest. If early stopping is used during
            fitting, n_estimators is the maximum number of trees.
        learning_rate : float
            The weight multiplying the output of each tree in the forest.
        max_samples : None or int or float
            The size of each bootstrapped subsample of the dataset. If None,
            then each subsample will have n_samples rows. If an int, each
            subsample will have max_samples rows. If a float between 0 and 1,
            each subsample will have int(max_samples * n_sample) rows.
        max_depth : int
            The max depth of each tree in the forest.
        max_features : None or int or float
            The number of features to consider when looking for the best split.
            See the documentation for sklearn.tree.DecisionTreeRegressor for
            more details.
        val_size : None or int or float
            Size of the validation set to use for early stopping. If None, no
            early stopping is performed. If an int, the validation set will have
            val_size rows. If a float, the validation set will have
            int(val_size * n_samples) rows. Note that if early stopping is used
            and max_samples is a float, the size of each subsample is calculated
            after the validation set has been set aside.
        threshold : float
            The error improvement that must be achieved every patience trees to
            avoid early stopping. Does nothing if val_size is None.
        patience : int
            The number of trees after which early stopping will be triggered if
            the error has not decreased by threshold. Does nothing if val_size
            is None.
        random_state : None or int or np.random.Generator
            The random state to use for subsampling and to pass to the trees in
            the forest. If None, a Generator is created with an unpredictable
            seed from the system. If an int, a Generator is created with
            random_state as its seed.
        verbose : bool
            Whether to print progress reports during fitting.

        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_samples = max_samples

        self.max_depth = max_depth
        self.max_features = max_features

        self.val_size = val_size
        self.threshold = threshold
        self.patience = patience

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit a boosted forest on the training set (X, y).

        If self.val_size is not None, then some of the provided samples will be
        set aside to use as a validation set for early stopping.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training inputs.
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The training targets.

        Returns
        -------
        self : BoostedForestRegressor
            The fitted regressor.

        """

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.flatten()

        random_state = self.random_state
        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        validate = self.val_size is not None
        if validate:
            X, X_va, y, y_va = train_test_split(
                X, y,
                test_size=self.val_size,
                random_state=random_state
            )

        n_samples = X.shape[0]
        n_bootstrap = get_n_bootstrap(n_samples, self.max_samples)

        current = np.zeros(y.shape)
        if validate:
            current_va = np.zeros(y_va.shape)
            lowest_error = get_error(y_va, current_va).mean()
            steps_since_improvement = 0

        estimators, stop = [], False
        for i in range(1, self.n_estimators + 1):
            weights = get_error(y, current)
            weights = weights / weights.sum()
            idx = random_state.choice(n_samples, size=n_bootstrap, p=weights)

            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=random_state
            ).fit(X[idx], y[idx] - current[idx])

            estimators.append(estimator)
            current += self.learning_rate * estimator.predict(X)
            message = f'Fit estimator {i}'

            if validate:
                current_va += self.learning_rate * estimator.predict(X_va)
                error = get_error(y_va, current_va).mean()
                message += f' - validation error is {error:.3f}'

                if error < lowest_error - self.threshold:
                    lowest_error = error
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if steps_since_improvement == self.patience:
                    message += '\nStopping early due to lack of improvement'
                    stop = True

            if self.verbose:
                print(message)

            if stop:
                break

        self.estimators_ = estimators
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """
        Makes a prediction with the fitted regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The inputs to make predictions for.

        Returns
        -------
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The output of the regressor.

        """

        check_is_fitted(self)
        X = check_array(X)

        outputs = [estimator.predict(X) for estimator in self.estimators_]

        return self.learning_rate * sum(outputs)

    def _more_tags(self):
        """Makes sure this class supports multioutput problems."""
        return {'multioutput' : True}
