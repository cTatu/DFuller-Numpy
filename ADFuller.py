#coding:utf-8
"""Augmented Dickey-Fuller test implemented using Numpy"""
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
import numpy as np

def narrow(input, dim, start, length):
    return np.take(input, range(start, start+length), axis=dim)

def ad_fuller(series, maxlag=None):
    """Get series and return the p-value and the t-stat of the coefficient"""
    if maxlag is None:
        n = int((len(series) - 1) ** (1./3))
    elif maxlag < 1:
        n = 1
    else:
        n = maxlag

    # Putting the X values on a Tensor with Double as type
    X = series

    # Generating the lagged tensor to calculate the difference
    X_1 = narrow(X, 0, 1, X.shape[0] - 1)

    # Re-sizing the x values to get the difference
    X = narrow(X, 0, 0, X.shape[0] - 1)
    dX = X_1 - X

    # Generating the lagged difference tensors
    # and concatenating the lagged tensors into a single one
    for i in range(1, n + 1):
        lagged_n = narrow(dX, 0, n - i, (dX.shape[0] - n))
        lagged_reshape = np.reshape(lagged_n, (lagged_n.shape[0], 1))
        if i == 1:
            lagged_tensors = lagged_reshape
        else:
            lagged_tensors = np.concatenate((lagged_tensors, lagged_reshape), 1)

    # Reshaping the X and the difference tensor
    # to match the dimension of the lagged ones
    X = narrow(X, 0, 0, X.shape[0] - n)
    dX = narrow(dX, 0, n, dX.shape[0] - n)
    dX = np.reshape(dX, (dX.shape[0], 1))

    # Concatenating the lagged tensors to the X one
    # and adding a column full of ones for the Linear Regression
    X = np.concatenate((np.reshape(X, (X.shape[0], 1)), lagged_tensors), 1)
    ones_columns = np.ones((X.shape[0], 1))
    X_ = np.concatenate((X, np.ones_like(ones_columns, dtype=np.float64)), 1)

    nobs = X_.shape[0]

    # Xb = y -> Xt.X.b = Xt.y -> b = (Xt.X)^-1.Xt.y
    coeff = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(X_.T, X_)), X_.T), dX)

    std_error = get_std_error(X_, dX, coeff)
    coeff_std_err = get_coeff_std_error(X_, std_error, coeff)[0]
    t_stat = (coeff[0]/coeff_std_err).item()

    p_value = mackinnonp(t_stat, regression="c", N=1)
    critvalues = mackinnoncrit(N=1, regression="c", nobs=nobs)
    critvalues = {
                  "1%" : critvalues[0],
                  "5%" : critvalues[1],
                  "10%" : critvalues[2]
                 }

    return t_stat, p_value, n, nobs, critvalues

def get_coeff_std_error(X, std_error, p):
    """Receive the regression standard error
    and calculate for the coefficient p"""
    std_coeff = []
    temp = np.linalg.inv(np.matmul(X.T, X))
    for i in range(len(p)):
        s = temp[i][i] * (std_error ** 2)
        s = np.sqrt(s)
        std_coeff.append(s)
    return std_coeff

def get_std_error(X, label, p):
    """Get the regression standard error"""
    std_error = 0
    y_new = np.matmul(X, p)
    std_error = np.sum((label[:, 0] - y_new[:, 0]) ** 2)
    return np.sqrt(std_error/X.shape[0])


def toeplitz(v):
    c = v.view(-1)
    vals = np.concatenate((c[::-1], c[1:]))
    a = np.expand_dims(np.arange(c.shape[0]), 0).T
    b = np.expand_dims(np.arange(c.shape[0] - 1, -1, step=-1))
    indx = a + b

    return vals[indx]

def toeplitz_like(x, n):
    r = x
    stop = x.shape[0] - 1

    if n < stop:
        stop = n
    else:
        stop = 2

    r = toeplitz(r)

    return narrow(narrow(r, 1, 0, stop), 0, stop - 1, r.shape[0] - stop)
