from sklearn.utils.estimator_checks import check_estimator

from mubofo import (
    MultioutputDecisionTree,
    MultioutputBoostedForest,
    MultioutputRandomForest
)

def test_MultioutputDecisionTree_is_estimator():
    check_estimator(MultioutputDecisionTree())

def test_MultioutputBoostedForest_is_estimator():
    check_estimator(MultioutputBoostedForest())

def test_MultioutputRandomForest_is_estimator():
    check_estimator(MultioutputRandomForest())