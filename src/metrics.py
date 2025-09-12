import numpy as np


def nse(y_true, y_pred):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) between observed and
    predicted values.

    The NSE is a normalized statistic that determines the relative magnitude
    of the residual variance compared to the measured data variance. It is
    commonly used to assess the predictive skill of hydrological models.

    Parameters
    ----------
    y_true : array-like
        Array of observed (true) values.
    y_pred : array-like
        Array of predicted values.

    Returns
    -------
    float
        Nash-Sutcliffe Efficiency coefficient. Values range from -∞ to 1,
        where 1 indicates a perfect match.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse_ = 1 - numerator / denominator
    return nse_


def nse_decomposed(y_true, y_pred):
    """
    Calculates the decomposed Nash-Sutcliffe Efficiency (NSE) and its
    components.

    The decomposed NSE provides insight into the performance of a hydrological
    model by breaking down the NSE into three components:
    - Alpha (α): Ratio of the standard deviations of predicted and observed
      values.
    - Beta (β): Bias, defined as the difference in means normalized by the
      standard deviation of observed values.
    - Corr (ρ): Correlation coefficient between observed and predicted values.

    Parameters
    ----------
    y_true : array-like
        Array of observed values.
    y_pred : array-like
        Array of predicted values.

    Returns
    -------
    nse : float
        Decomposed Nash-Sutcliffe Efficiency value.
    alpha : float
        Ratio of standard deviations (std(y_pred) / std(y_true)).
    beta : float
        Normalized bias ((mean(y_pred) - mean(y_true)) / std(y_true)).
    corr : float
        Correlation coefficient between y_true and y_pred.

    References
    ----------
    - Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009).
    Decomposition of the mean squared error and NSE performance criteria:
    Implications for improving hydrological modelling. Journal of Hydrology,
    377(1-2), 80-91.
    """
    alpha = np.std(y_pred) / np.std(y_true)
    beta = (np.mean(y_pred) - np.mean(y_true)) / np.std(y_true)
    xy = np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)))
    corr = xy / ((len(y_true)) * np.std(y_true) * np.std(y_pred))
    nse_ = 2 * alpha * corr - alpha**2 - beta**2
    return nse_, alpha, beta, corr


def kge(y_true, y_pred):
    """
    Calculates the Kling-Gupta Efficiency (KGE) between observed and predicted
    values.

    The KGE metric combines correlation, bias, and variability to assess
    the accuracy of hydrological models.

    It is defined as:
        KGE = 1 - sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    where:
        corr  = Pearson correlation coefficient between y_true and y_pred
        alpha = Ratio of standard deviations (y_pred_std / y_true_std)
        beta  = Ratio of means (y_pred_mean / y_true_mean)

    NaN values in y_true are ignored in the calculation.

    Parameters
    ----------
    y_true : array-like
        Observed values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    kge : float
        Kling-Gupta Efficiency value.
    corr : float
        Pearson correlation coefficient.
    alpha : float
        Ratio of standard deviations.
    beta : float
        Ratio of means.
    """

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
    kge_ = 1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge_, corr, alpha, beta


def fdc(y, low_fraction, high_fraction):
    """
    Calculates a segment of the Flow Duration Curve (FDC) for a given array
    of flow values.

    Parameters
    ----------
    y : array-like
        Array of flow values.
    low_fraction : float
        Lower fraction (between 0 and 1) of the sorted flow values to start
        the FDC segment.
    high_fraction : float
        Upper fraction (between 0 and 1) of the sorted flow values to end
        the FDC segment.

    Returns
    -------
    fdc : ndarray
        Array containing the flow values between the specified low and high
        fractions of the sorted data.

    Notes
    -----
    The input array `y` is sorted in ascending order before extracting
    the segment.
    """
    y = np.sort(y)
    n = len(y)
    fdc_ = y[int(low_fraction * n) : int(high_fraction * n)]
    return fdc_


def fhv(y_true, y_pred, limit=0.02):
    """
    Calculate the %Bias Flow High-segment Volume (FHV) between observed and
    predicted values.

    This metric quantifies the bias in the high-flow segment of the flow
    duration curve (FDC) between the predicted and observed data. It is
    commonly used in hydrological model evaluation.

    Parameters
    ----------
    y_true : array-like
        Observed flow values.
    y_pred : array-like
        Predicted flow values.
    limit : float, optional
        Fraction of the flow duration curve to consider as the high-flow
        segment (default is 0.02).

    Returns
    -------
    fhv : float
        Percent bias in the high-segment volume between predicted and observed
        flows.

    Notes
    -----
    Requires the `fdc` function to compute the flow duration curve segment.
    """
    y_true = fdc(y_true, limit, 1)
    y_pred = fdc(y_pred, limit, 1)

    fhv_ = 100 * (np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)
    return fhv_


def flv(y_true, y_pred, limit=0.3, fix_low=False):
    """
    Calculate the %Bias Flow Low-segment Volume (FLV) between observed and
    predicted flow data.

    This metric evaluates the bias in the low-flow segment of the flow duration
    curve (FDC) between the true and predicted values. The function applies a
    limit to select the low-flow segment and optionally fixes zero values to a
    small positive number to avoid issues with logarithms.

    Parameters
    ----------
    y_true : array-like
        Observed (true) flow values.
    y_pred : array-like
        Predicted flow values.
    limit : float, optional
        Upper limit (as a fraction of exceedance probability) for the low-flow
        segment (default is 0.3).
    fix_low : bool, optional
        If True, replaces zero values in the low-flow segment with a small
        positive number (1e-6) to avoid issues with logarithmic calculations
        (default is False).

    Returns
    -------
    flv : float
        The %Bias Flow Low-segment Volume (FLV) value.
    """
    y_true = fdc(y_true, 0, limit)
    y_pred = fdc(y_pred, 0, limit)

    if fix_low:
        y_true[y_true == 0] = 1e-6
        y_pred[y_pred == 0] = 1e-6

    log_y_true = np.log(y_true)
    log_y_pred = np.log(y_pred)

    flv_ = (
        -1
        * 100
        * (
            np.sum(np.abs(log_y_pred - np.min(log_y_pred)))
            - np.sum(np.abs(log_y_true - np.min(log_y_true)))
        )
        / np.sum(np.abs(log_y_true - np.min(log_y_true)))
    )
    return flv_


def fmv(y_true, y_pred, lower_limit=0.2, upper_limit=0.7, fix_low=False):
    """
    Calculate the percent bias in Flow Mid-segment Volume (FMV) between
    observed and predicted data.

    This metric compares the change in flow volume between the start and end
    of the mid-segment of the flow duration curve (FDC) for both observed
    (`y_true`) and predicted (`y_pred`) values.

    Parameters
    ----------
        y_true : array-like
            Observed flow values.
        y_pred : array-like
            Predicted flow values.
        lower_limit : float, optional
            Lower quantile limit for the FDC segment (default is 0.2).
        upper_limit : float, optional
            Upper quantile limit for the FDC segment (default is 0.7).
        fix_low : bool, optional
            If True, replaces zero values in the FDC segment with a small
            positive value (1e-6) to avoid log(0) errors (default is False).

    Returns
    -------
        fmv : float
            Percent bias in FMV between predicted and observed data.

    Notes
    -----
        - The function uses the `fdc` function to extract the mid-segment of
          the flow duration curve.
        - The calculation is based on the logarithmic difference between the
          start and end of the FDC segment.
        - If `fix_low` is True, zero values are replaced to prevent
          mathematical errors.
    """
    y_true = fdc(y_true, lower_limit, upper_limit)
    y_pred = fdc(y_pred, lower_limit, upper_limit)

    if fix_low:
        y_true[y_true == 0] = 1e-6
        y_pred[y_pred == 0] = 1e-6

    fmv_ = (
        100
        * (
            np.abs(np.log(y_pred[0]) - np.log(y_pred[-1]))
            - np.abs(np.log(y_true[0]) - np.log(y_true[-1]))
        )
        / np.abs(np.log(y_true[0]) - np.log(y_true[-1]))
    )
    return fmv_


def mse(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) between the true and predicted
    values.

    Parameters
    ----------
        y_true : array-like
            Array of true target values.
        y_pred : array-like
            Array of predicted values.

    Returns
    -------
        mse : float
            The mean squared error between y_true and y_pred.
    """
    mse_ = np.mean((y_true - y_pred) ** 2)
    return mse_


def mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between the true and predicted
    values.

    Parameters
    ----------
        y_true : array-like
            Array of true target values.
        y_pred : array-like
            Array of predicted values.

    Returns
    -------
        mae : float
            The mean absolute error between y_true and y_pred.
    """
    mae_ = np.mean(np.abs(y_true - y_pred))
    return mae_


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted
    values.

    Parameters
    ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated target values.

    Returns
    -------
        rmse : float
            The RMSE value.
    """
    rmse_ = np.sqrt(mse(y_true, y_pred))
    return rmse_
