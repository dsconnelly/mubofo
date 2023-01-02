import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from mubofo import (
    MultioutputDecisionTree,
    MultioutputBoostedForest,
    MultioutputRandomForest
)

def test_MultioutputDecisionTree_predictions(two_dimensional_data):
    X, y = two_dimensional_data
    base = DecisionTreeRegressor(max_depth=3, random_state=123).fit(X, y)
    model = MultioutputDecisionTree(max_depth=3, random_state=123).fit(X, y)

    assert np.allclose(base.predict(X), model.predict(X))

def test_MultioutputBoostedForest_predictions(two_dimensional_data):
    X, y = two_dimensional_data
    model = MultioutputBoostedForest(max_depth=3, random_state=123).fit(X, y)
    score = r2_score(y, model.predict(X))

    assert score > 0.9

def test_MultioutputRandomForest_predictions(two_dimensional_data):
    X, y = two_dimensional_data
    base = RandomForestRegressor(max_depth=3, random_state=123).fit(X, y)
    model = MultioutputRandomForest(max_depth=3, random_state=123).fit(X, y)

    assert np.allclose(base.predict(X), model.predict(X))

