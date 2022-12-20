import numpy as np

from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from mubofo import MultioutputDecisionTree

def test_is_estimator():
    check_estimator(MultioutputDecisionTree())

def test_predictions():
    X, Y = make_regression(n_samples=200, n_features=40, n_targets=5)
    base = DecisionTreeRegressor(max_depth=3, random_state=123).fit(X, Y)
    model = MultioutputDecisionTree(max_depth=3, random_state=123).fit(X, Y)

    assert np.allclose(base.predict(X), model.predict(X))

def test_1d_importances():
    X, Y = make_regression(n_samples=200, n_features=40, n_targets=1)
    base = DecisionTreeRegressor(max_depth=3, random_state=123).fit(X, Y)
    model = MultioutputDecisionTree(max_depth=3, random_state=123).fit(X, Y)

    assert np.allclose(base.feature_importances_, model.feature_importances_)
