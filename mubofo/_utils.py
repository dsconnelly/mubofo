import numpy as np

def get_error(y, current, squared=False):
    """
    Computes error between targets and model predictions at current iteration.

    The error is calculated as the (root) mean square error over output channels
    if there is more than one, depending on the squared parameter.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,) or (n_samples, n_features)
        The training or validation targets.
    current : np.ndarray of shape y.shape
        The model predictions at the current training iteration.
    squared : bool
        Whether or not to return squared error.

    Returns
    -------
    error : np.ndarray of shape (n_samples,)
        The (root) mean square error over output channels.

    """

    error = (y - current) ** 2
    if error.ndim > 1:
        error = error.mean(axis=1)

    if squared:
        return error

    return np.sqrt(error)

def get_n_bootstrap(n_samples, max_samples):
    """
    Computes the size of bootstrapped subsamples.

    Parameters
    ----------
    n_samples : int
        The number of provided training samples
    max_samples : None or int or float
        See the docstring for BoostedForestRegressor.__init__ for more details
        on how max_samples defines the size of subsamples.

    Returns
    -------
    n_bootstrap : int
        The size of each bootstrapped subsample.

    """

    if max_samples is None:
        return n_samples

    if isinstance(max_samples, int):
        return max_samples

    if isinstance(max_samples, float):
        return max(round(max_samples * n_samples), 1)