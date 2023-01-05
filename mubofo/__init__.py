"""Boosted forest regression with native support for multioutput problems."""

__version__ = '0.2.8'

from ._forests import MultioutputBoostedForest, MultioutputRandomForest
from ._trees import MultioutputDecisionTree

__all__ = [
    'MultioutputBoostedForest', 
    'MultioutputDecisionTree',
    'MultioutputRandomForest'
]
