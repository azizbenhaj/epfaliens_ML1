import numpy as np
from utils import sigmoid


# Mean Square error function
def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the mse loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


# negative log-likelihood loss for the binary classification problem
def compute_neg_log_likelihood_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    """
    # Check dim
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # Compute the predicted output
    pred = sigmoid(tx @ w)

    # Ensure numerical stability by avoiding log(0) and log(1)
    pred[pred == 1] -= 1e-9
    pred[pred == 0] += 1e-9

    # Compute the negative log likelihood loss
    return (-1 / y.shape[0]) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))


# Accuracy fuction
def compute_acc(y_true, y_pred):
    """
    Compute the accuracy of a classification model.

    Parameters:
    - y_true (numpy.ndarray): The true target labels.
    - y_pred (numpy.ndarray): The predicted labels.

    Returns:
    The accuracy as a percentage (0 to 100).
    """
    return 100 * np.sum(y_pred == y_true) / y_true.shape[0]


# F1 score function
def compute_f1_score(y_true, y_pred):
    """
    Compute the F1 score, a measure of a classification model's accuracy that balances precision and recall.

    Parameters:
    - y_true (list or numpy.ndarray): The true target labels.
    - y_pred (list or numpy.ndarray): The predicted labels.

    Returns:
    The F1 score, a value between 0 and 1, where a higher score indicates better model performance.
    """

    # Calculate true positives, false positives, and false negatives
    tp = sum((a == 1 and p == 1) for a, p in zip(y_true, y_pred))
    fp = sum((a == 0 and p == 1) for a, p in zip(y_true, y_pred))
    fn = sum((a == 1 and p == 0) for a, p in zip(y_true, y_pred))

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1
