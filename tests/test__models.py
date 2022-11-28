from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.utils.estimator_checks import check_estimator

from mubofo import BoostedForestRegressor

def test_is_estimator():
    check_estimator(BoostedForestRegressor())

def test_learns_well():
    X, Y = make_regression(n_samples=10000, n_features=40, n_targets=20)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    model = BoostedForestRegressor(max_depth=2).fit(X, Y)
    score = r2_score(Y, model.predict(X))

    assert score > 0.9
