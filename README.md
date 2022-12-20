# mubofo

mubofo (from **mu**ltioutput **bo**osted **fo**rest) is a Python implementation of boosted forest regression with native support for multioutput problems.

[Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning)) is a technique for building an ensemble of weak learners (often tree models) wherein each new regressor is trained on the residual errors accrued by the regressors trained so far. However, most popular implementations use gradient boosting, a mathematical reformulation that involves Taylor-expanding the loss function about the current prediction at each iteration and is thus impractical when the output space is multidimensional.

Some gradient boosting libraries support multioutput problems by training one-dimensional models for each output channel, but such an approach risks missing correlations between outputs and is much slower than a single multioutput model would be.

mubofo solves this problem by returning to the "train on the residuals" perspective of boosting. The main class is `MultioutputBoostedForest`, which implements the [scikit-learn estimator API](https://scikit-learn.org/stable/developers/develop.html). While implemented entirely in Python, `MultioutputBoostedForest` is fast enough for practical use because the individual trees in the forest are (wrappers of) [`sklearn.tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) instances, and so most model fitting is handled by the C backend.

mubofo can be installed with `pip install mubofo`.
