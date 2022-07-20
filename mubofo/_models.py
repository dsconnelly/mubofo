import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class BoostedForestRegressor(BaseEstimator, RegressorMixin):
    """
    Boosted forest for (possibly multioutput) regression problems.

    The BoostedForestRegressor class implements the scikit-learn estimator API
    and can therefore be used like any other scikit-learn model.

    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.2,
        max_depth=5,
        max_samples=None,
        max_features=None,
        n_validation=None,
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
        max_depth : int
            The max depth of each tree in the forest.
        max_samples : None or int or float
            The size of each bootstrapped subsample of the dataset. If None,
            then each subsample will have n_samples rows. If an int, each
            subsample will have max_samples rows. If a float between 0 and 1,
            each subsample will have int(max_samples * n_sample) rows.
        max_features : None or int or float
            The number of features to consider when looking for the best split.
            See the documentation for sklearn.tree.DecisionTreeRegressor for
            more details.
        n_validation : None or int or float
            Size of the validation set to use for early stopping. If None, no
            early stopping is performed. If an int, the validation set will have
            n_validation rows. If a float, the validation set will have
            int(n_validation * n_samples) rows. Note that if early stopping is
            used and max_samples is a float, the size of each subsample is
            calculated after the validation set has been set aside.
        threshold : float
            The error improvement that must be achieved every patience trees to
            avoid early stopping. Does nothing if n_validation is None.
        patience : int
            The number of trees after which early stopping will be triggered if
            the error has not decreased by threshold. Does nothing if
            n_validation is None.
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

        self.max_depth = max_depth
        self.max_samples = max_samples
        self.max_features = max_features

        self.n_validation = n_validation
        self.threshold = threshold
        self.patience = patience

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit a boosted forest on the training set (X, y).

        If self.n_validation is not None, then some of the provided samples will
        be set aside to use as a validation set for early stopping.

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

        early_stopping = self.n_validation is not None
        if early_stopping:
            n_validation = self.n_validation
            if isinstance(n_validation, float):
                n_validation = int(n_validation * X.shape[0])

            perm = random_state.permutation(X.shape[0])
            idx_tr, idx_va = perm[:-n_validation], perm[-n_validation:]
            X, y, X_va, y_va = X[idx_tr], y[idx_tr], X[idx_va], y[idx_va]

        n_samples = X.shape[0]
        if self.max_samples is None:
            max_samples = n_samples
        elif isinstance(self.max_samples, int):
            max_samples = self.max_samples
        elif isinstance(self.max_samples, float):
            max_samples = max(int(self.max_samples * n_samples), 1)

        current = np.zeros(y.shape)
        if early_stopping:
            current_va = np.zeros(y_va.shape)
            lowest_error = np.sqrt((y_va ** 2).mean())
            trees_since_improvement = 0

        trees = [DecisionTreeRegressor(
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=random_state
        ) for _ in range(self.n_estimators)]

        for j, tree in enumerate(trees, start=1):
            errors = (current - y) ** 2
            if errors.ndim > 1:
                errors = errors.mean(axis=1)

            weights = errors / errors.sum()
            idx = random_state.choice(n_samples, size=max_samples, p=weights)
            tree.fit(X[idx], y[idx] - current[idx])

            message = f'Fit tree {j} out of {self.n_estimators}'
            current += self.learning_rate * tree.predict(X)

            stop = False
            if early_stopping:
                current_va += self.learning_rate * tree.predict(X_va)
                error_va = np.sqrt(((current_va - y_va) ** 2).mean())
                message += f' -- validation error is {error_va:.3f}'

                if error_va < lowest_error - self.threshold:
                    lowest_error = error_va
                    trees_since_improvement = 0
                else:
                    trees_since_improvement += 1

                if trees_since_improvement == self.patience:
                    message += '\nStopping early due to lack of improvement.'
                    stop = True

            if self.verbose:
                print(message)

            if stop:
                break

        self.estimators_ = trees[:j]
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

        outputs = [tree.predict(X) for tree in self.estimators_]

        return self.learning_rate * sum(outputs)

    def _more_tags(self):
        """Makes sure this class supports multioutput problems."""
        return {'multioutput' : True}
