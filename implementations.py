import numpy as np
from utils import *
from cost import compute_neg_log_likelihood_loss, compute_mse


# Linear regression using gradient descent --------------------------------------------------------
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        last_w: the model parameter for the last iteration of GD
        mse: the loss value for the last iteration of GD
    """

    w = initial_w
    for _ in range(max_iters):
        grad, _ = compute_Linear_regression_gradient(y, tx, w)
        w = w - gamma * grad

    last_w = w
    mse = compute_mse(y, tx, last_w)
    return last_w, mse


# Linear regression using stochastic gradient descent  ----------------------------------
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        last_w: the model parameter for the last iteration of SGD
        mse: the loss value for the last iteration of SGD
    """
    w = initial_w
    for _ in range(max_iters):
        grad, _ = compute_Linear_regression_gradient(y, tx, w)
        w = w - gamma * grad

    last_w = w
    mse = compute_mse(y, tx, last_w)
    return last_w, mse


# Least squares regression using normal equations -----------------------------------------
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w)
    return w, mse


# Ridge regression using normal equations -------------------------------------------------
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w)
    return w, mse


# Logistic regression using gradient descent or SGD (y ∈ {0, 1}) ----------------------------
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Implement the logistic regression using gradient descent
    Return the loss and final w.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """

    w = initial_w

    # start the logistic regression
    for _ in range(max_iters):
        # get loss and update w.
        grad = compute_logistic_regression_grad(y, tx, w)
        w -= gamma * grad

    neg_log_likelihood_loss = compute_neg_log_likelihood_loss(y, tx, w)

    return w, neg_log_likelihood_loss


# Regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with regularization term λ∥w∥2) ---------
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Implement the penalized logistic regression using gradient descent
    Return the loss and final w.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        grad = compute_penalized_logistic_regression_grad(y, tx, w, lambda_)
        w -= gamma * grad

    neg_log_likelihood_loss = compute_neg_log_likelihood_loss(y, tx, w)
    return w, neg_log_likelihood_loss


# generalized regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with l1 or l2 reg) ---------
def reg_logistic_regression_adaptive_lr(
    y, tx, lambda_, initial_w, max_iters, gamma, order=2, stoch=True, adapt_lr=True
):
    """
    Implement the penalized logistic regression using gradient descent
    Return the loss and final w.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        order (int, optional): The order of the logistic regression  regulizer function (default is 2).
        stoch (bool, optional): If True, use stochastic gradient descent. If False, use gradient descent (default is True).
        adapt_lr (bool, optional): If True, adapt the learning rate during training (default is True).
    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """

    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        grad = 0

        # compute stochastic gradient descent
        if stoch:
            for y_batch, tx_batch in batch_iter(y, tx, batch_size=32, num_batches=1):
                grad = compute_penalized_logistic_regression_grad(
                    y_batch, tx_batch, w, lambda_, order=order
                )

        # compute full gradient descent
        else:
            grad = compute_penalized_logistic_regression_grad(
                y, tx, w, lambda_, order=order
            )

        w = w - gamma / (iter % 50 + 1) * grad if adapt_lr else w - gamma * grad

    # compute the negative log likelihood loss
    neg_log_likelihood_loss = compute_neg_log_likelihood_loss(y, tx, w)

    return w, neg_log_likelihood_loss
