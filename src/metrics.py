import numpy as np


def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency (NSE)"""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - numerator / denominator
    return nse


def nse_decomposed(y_true, y_pred):
    """Nash-Sutcliffe Efficiency (NSE) decomposed"""
    alpha = np.std(y_pred) / np.std(y_true)
    beta = (np.mean(y_pred) - np.mean(y_true)) / np.std(y_true)
    xy = np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)))
    corr = xy / ((len(y_true)) * np.std(y_true) * np.std(y_pred))
    nse = 2 * alpha * corr - alpha**2 - beta**2
    return nse, alpha, beta, corr


def kge(y_true, y_pred):
    """Kling-Gupta Efficiency (KGE)"""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = y_pred[~np.isnan(y_true)]
    y_true = y_true[~np.isnan(y_true)]

    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    y_true_std = np.std(y_true)
    y_pred_std = np.std(y_pred)

    xy = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    corr = xy / ((len(y_true)) * y_true_std * y_pred_std)

    alpha = y_pred_std / y_true_std
    beta = y_pred_mean / y_true_mean
    kge = 1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge, corr, alpha, beta


def fdc(y, low_fraction, high_fraction):
    """Flow Duration Curve (FDC) fraction"""
    y = np.sort(y)
    n = len(y)
    fdc = y[int(low_fraction * n) : int(high_fraction * n)]
    return fdc


def fhv(y_true, y_pred, limit=0.02):
    """%Bias Flow High-segment volume (FHV)"""
    y_true = fdc(y_true, limit, 1)
    y_pred = fdc(y_pred, limit, 1)

    fhv = 100 * (np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)
    return fhv


def flv(y_true, y_pred, limit=0.3, fix_low=False):
    """%Bias Flow Low-segment volume (FLV)"""
    y_true = fdc(y_true, 0, limit)
    y_pred = fdc(y_pred, 0, limit)

    if fix_low:
        y_true[y_true == 0] = 1e-6
        y_pred[y_pred == 0] = 1e-6

    log_y_true = np.log(y_true)
    log_y_pred = np.log(y_pred)

    flv = (
        -1
        * 100
        * (
            np.sum(np.abs(log_y_pred - np.min(log_y_pred)))
            - np.sum(np.abs(log_y_true - np.min(log_y_true)))
        )
        / np.sum(np.abs(log_y_true - np.min(log_y_true)))
    )
    return flv


def fmv(y_true, y_pred, lower_limit=0.2, upper_limit=0.7, fix_low=False):
    """%Bias Flow Mid-segment volume (FLV)"""
    y_true = fdc(y_true, lower_limit, upper_limit)
    y_pred = fdc(y_pred, lower_limit, upper_limit)

    if fix_low:
        y_true[y_true == 0] = 1e-6
        y_pred[y_pred == 0] = 1e-6

    fmv = (
        100
        * (
            np.abs(np.log(y_pred[0]) - np.log(y_pred[-1]))
            - np.abs(np.log(y_true[0]) - np.log(y_true[-1]))
        )
        / np.abs(np.log(y_true[0]) - np.log(y_true[-1]))
    )
    return fmv


def mse(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) between the true and predicted values.

    Parameters:
        y_true (array-like): Array of true target values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: The mean squared error between y_true and y_pred.

    Example:
        >>> mse(np.array([1, 2, 3]), np.array([1, 2, 4]))
        0.3333333333333333
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between the true and predicted values.

    Parameters:
        y_true (array-like): Array of true target values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: The mean absolute error between y_true and y_pred.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The RMSE value.

    Notes:
        This function assumes that `mse` (Mean Squared Error) and `np` (NumPy) are already defined/imported.
    """
    rmse = np.sqrt(mse(y_true, y_pred))
    return rmse
