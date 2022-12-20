import numpy as np

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils.estimator_checks import check_estimator

from mubofo import MultioutputBoostedForest, MultioutputRandomForest

def test_MultioutputBoostedForest_is_estimator():
    check_estimator(MultioutputBoostedForest())

def test_MultioutputBoostedForest_learns_well():
    X, Y = make_regression(n_samples=10000, n_features=40, n_targets=20)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    model = MultioutputBoostedForest(max_depth=2).fit(X, Y)
    score = r2_score(Y, model.predict(X))

    assert score > 0.9

def test_MultioutputRandomForest_is_estimator():
    check_estimator(MultioutputRandomForest())

def test_MultioutputRandomForest_predictions():
    X, Y = make_regression(n_samples=150, n_features=40, n_targets=10)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    base = RandomForestRegressor(max_depth=3, random_state=123).fit(X, Y)
    model = MultioutputRandomForest(max_depth=3, random_state=123).fit(X, Y)

    assert np.allclose(base.predict(X), model.predict(X))

def test_feature_importances__shape():
    X, Y = make_regression(n_samples=150, n_features=40, n_targets=10)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    random = MultioutputRandomForest(max_depth=3).fit(X, Y)
    boosted = MultioutputBoostedForest(max_depth=3).fit(X, Y)

    assert random.feature_importances_.shape == (40, 10)
    assert boosted.feature_importances_.shape == (40, 10)

def test_feature_importances__sum():
    X, Y = make_regression(n_samples=150, n_features=40, n_targets=10)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    random = MultioutputRandomForest(max_depth=3).fit(X, Y)
    boosted = MultioutputBoostedForest(max_depth=3).fit(X, Y)

    assert np.allclose(np.ones(10), random.feature_importances_.sum(axis=0))
    assert np.allclose(np.ones(10), boosted.feature_importances_.sum(axis=0))