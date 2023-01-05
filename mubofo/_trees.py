from __future__ import annotations

from typing import Optional

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, _check_sample_weight

class MultioutputDecisionTree(DecisionTreeRegressor):
    """Tree regressor with per-target feature importances."""

    feature_importances_ = np.array([])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray]=None,
        **_
    ) -> MultioutputDecisionTree:
        """
        Fit the regressor and calculate feature importances and absolute means.

        Parameters
        ----------
        X : Training inputs.
        Y : Training targets.
        sample_weight : Sample weights. If None, samples are weighted equally.

        Returns
        -------
        MultioutputTreeRegressor
            The fitted regressor.

        """

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        weights = _check_sample_weight(sample_weight, X)
        super().fit(X, y, weights, check_input=False)

        idx = weights > 0
        X, y, weights = X[idx], y[idx], weights[idx]

        if y.ndim > 1:
            weights = weights.reshape(-1, 1)

        n_samples = self.tree_.weighted_n_node_samples
        impurities = np.zeros((len(n_samples), self.n_outputs_))
        self._get_impurities(X.astype(np.float64), y, weights, impurities, 0)
  
        importances = np.zeros((X.shape[1], self.n_outputs_))
        for node in range(len(n_samples)):
            left = self.tree_.children_left[node]
            right = self.tree_.children_right[node]

            if left == right:
                continue

            gain = n_samples[node] * impurities[node]
            gain -= n_samples[left] * impurities[left]
            gain -= n_samples[right] * impurities[right]

            feature = self.tree_.feature[node]
            importances[feature] += gain

        leaves = self.tree_.children_left == self.tree_.children_right
        outputs = abs(self.tree_.value[leaves, :, 0].T)
        n_samples = n_samples[leaves]

        sums = importances.sum(axis=0)
        importances[:, sums != 0] /= sums[sums != 0]

        if self.n_outputs_ == 1:
            importances = importances.flatten()

        self.abs_means_: np.ndarray
        self.feature_importances_ = importances
        self.abs_means_ = (n_samples * outputs).sum(axis=1) / n_samples.sum()

        return self

    def _get_impurities(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        impurities: np.ndarray,
        node: int
    ) -> None:
        """
        Calculate per-output impurities at a given node and recurse.

        Parameters
        ----------
        X : Training inputs reaching `node`.
        y : Training targets reaching `node`.
        weights : Training `sample_weight` corresponding to the inputs and
            targets passed as `X` and `y`, respectively.
        impurities : Partially-filled array where impurities will be stored.
        node : Node at which impurities will be calculated.

        """

        if not len(X):
            impurities[node] = 0
            return

        else:
            W = weights.sum()
            mean = (weights * y).sum(axis=0) / W
            impurities[node] = (weights * (y - mean) ** 2).sum(axis=0) / W

        left = self.tree_.children_left[node]
        right = self.tree_.children_right[node]

        if left == right:
            return

        idx = X[:, self.tree_.feature[node]] <= self.tree_.threshold[node]
        self._get_impurities(X[idx], y[idx], weights[idx], impurities, left)
        self._get_impurities(X[~idx], y[~idx], weights[~idx], impurities, right)
