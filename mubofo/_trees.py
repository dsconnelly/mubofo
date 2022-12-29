from __future__ import annotations

from typing import Optional

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import _check_sample_weight, check_X_y

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
        X : np.ndarray of shape (n_samples, n_features)
            The training inputs.
        Y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The training targets.
        sample_weight : np.ndarray of shape (n_samples,) or None
            The sample weights. If None, all samples are weighted equally.

        Returns
        -------
        self : MultioutputTreeRegressor
            The fitted regressor.

        """

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        tree = super().fit(X, y, sample_weight, check_input=False).tree_
        
        paths = self.decision_path(X).tocsc().astype(bool)
        n_samples = paths.sum(axis=0).getA1()
        n_nodes = len(n_samples)

        impurities = np.zeros((n_nodes, self.n_outputs_))
        for node in range(n_nodes):
            idx = np.squeeze(paths[:, node].toarray())
            impurities[node] = y[idx].var(axis=0)

        importances = np.zeros((X.shape[1], self.n_outputs_))
        for node in range(n_nodes):
            left = tree.children_left[node]
            right = tree.children_right[node]

            if left == right:
                continue

            gain = n_samples[node] * impurities[node]
            gain -= n_samples[left] * impurities[left]
            gain -= n_samples[right] * impurities[right]

            feature = tree.feature[node]
            importances[feature] += gain

        leaves = tree.children_left == tree.children_right
        outputs = abs(tree.value[leaves, :, 0].T)
        n_samples = n_samples[leaves]

        sums = importances.sum(axis=0)
        importances[:, sums != 0] /= sums[sums != 0]

        if self.n_outputs_ == 1:
            importances = importances.flatten()

        self.abs_means_: np.ndarray
        self.feature_importances_ = importances
        self.abs_means_ = (n_samples * outputs).sum(axis=1) / n_samples.sum()

        return self
