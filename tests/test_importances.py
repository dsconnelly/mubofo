import numpy as np

from sklearn.tree import DecisionTreeRegressor

from mubofo import (
    MultioutputDecisionTree,
    MultioutputBoostedForest,
    MultioutputRandomForest
)

def test_MultioutputDecisionTree_importances(one_dimensional_data):
    X, y = one_dimensional_data
    sample_weight = np.random.choice(3, size=X.shape[0])

    base = DecisionTreeRegressor(random_state=123).fit(X, y, sample_weight)
    model = MultioutputDecisionTree(random_state=123).fit(X, y, sample_weight)

    assert np.allclose(base.feature_importances_, model.feature_importances_)

def test_MultioutputRandomForest_importances(two_dimensional_data):
    X, y = two_dimensional_data
    model = MultioutputRandomForest(random_state=123).fit(X, y)
    importances = model.feature_importances_

    assert np.all(importances >= 0)
    assert np.allclose(importances.sum(axis=0), 1)

def test_MultioutputBoostedForest_importances(two_dimensional_data):
    X, y = two_dimensional_data
    model = MultioutputBoostedForest(random_state=123).fit(X, y)
    importances = model.feature_importances_

    assert np.all(importances >= 0)
    assert np.allclose(importances.sum(axis=0), 1)