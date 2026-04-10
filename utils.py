import numpy as np


def print_cross_validation_results(res, functions):
    """
    Print cross-validation results, including hyperparameters and performance metrics.

    Args:
        res: A numpy array containing results (e.g., F1 score, accuracy, hyperparameters).
        functions: A list of function names.
    """
    for i in range(len(res)):
        print(
            f"Using {functions[i]}: lambda = {res[i,2]}, max_iters= {res[i,3]},gamma = {res[i,4]} using L{res[i,5]:.0f} : F1 = {res[i,0]:.2f} and acc = {res[i,1]:.2f} "
        )


# gradient
def compute_Linear_regression_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        grad: numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
        err: error array of shape=(N, )
    """

    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


# logistic regression gradient
def compute_logistic_regression_grad(y, tx, w):
    """compute the gradient of the logistic regression loss.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D)

    Returns:
        a vector of shape (D)
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) * (1 / y.shape[0])


# penalized logistic regression gradient
def compute_penalized_logistic_regression_grad(y, tx, w, lambda_, order=2):
    """return the gradient of penalized logistic regression function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar
        order: Order of regularization (default is 2)

    Returns:
        gradient: shape=(D, 1)
    """
    return compute_logistic_regression_grad(y, tx, w) + order * lambda_ * (
        w ** (order - 1)
    )


# sigmoid function
def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-t))


# pred form proba to {-1,1}
def classifictaion_pred(pred, using0_class=True, model_name="logistic_regression"):
    """
    Perform binary classification prediction by converting probabilities to {-1, 1} labels.

    Args:
        pred: Predicted probabilities
        using0_class: Whether to use 0 as a class label (default is True)
        model_name: Name of the classification model (default is "logistic_regression")

    Returns:
        pred: Predicted labels (-1 or 1) based on the model
    """
    threshold = (
        0 if model_name == "logistic_regression" else (np.max(pred) - np.mean(pred)) / 2
    )

    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0 if using0_class else -1

    return pred


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    This function takes as input two iterables, the output desired values 'y' and the input data 'tx'.
    It outputs an iterator that provides mini-batches of `batch_size` matching elements from `y` and `tx`.
    The data can be randomly shuffled to avoid the original data's ordering affecting the randomness of
    the minibatches.

    Parameters:
        y (numpy.ndarray): Target values.
        tx (numpy.ndarray): Input data.
        batch_size (int): Size of each minibatch.
        num_batches (int): Number of minibatches to generate (default is 1).
        shuffle (bool): Whether to shuffle the data (default is True).

    Returns:
        batch_y (numpy.ndarray): Mini-batch of target values.
        batch_tx (numpy.ndarray): Mini-batch of input data.

    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """
    Perform feature expansion of x to a specified degree.

    Args:
        x: Input data
        degree: Degree of feature expansion

    Returns:
        Polynomial feature matrix
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly
