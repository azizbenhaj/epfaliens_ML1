import numpy as np
from cost import *
from utils import classifictaion_pred
from implementations import *
import pickle


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)

    # suffle indices
    indices = np.random.permutation(num_row)

    # build K intervals according to the suffled indices
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)


def cross_validation_step(
    y, x, k_indices, k, f, initial_w, max_iters, gamma, lambda_=0, order=2
):
    """return the loss of penalized ridge regression for a fold corresponding to k_indices

    Args:
    - y (numpy.ndarray): The target labels, shape=(N,).
    - x (numpy.ndarray): The feature matrix, shape=(N,).
    - k_indices (list of numpy.ndarray): 2D array returned by build_k_indices().
    - k (int): The index of the current fold (not to be confused with k_fold, which is the number of folds).
    - f (function): A function for performing ridge regression or other related tasks.
    - initial_w (numpy.ndarray): The initial weights for the regression model.
    - max_iters (int): The maximum number of iterations for the optimization.
    - gamma (float): The learning rate.
    - lambda_ (float, optional): The regularization parameter (default is 0).
    - order (int, optional): The order for polynomial expansion (default is 2).

    Returns:
    - loss_te (float): The testing loss.
    - f1_score (float): The F1 score for classification.
    - acc (float): The accuracy for classification.

    """

    # Create masks for training and testing data based on fold k
    train_mask = np.ones(y.shape[0], dtype=bool)
    train_mask[k_indices[k]] = 0
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test = x[~train_mask]
    y_test = y[~train_mask]

    # compute the model parameter
    w = 0
    if f == "logistic_regression":
        w, _ = reg_logistic_regression_adaptive_lr(
            y_train, x_train, lambda_, initial_w, max_iters, gamma, order=order
        )

    elif f == "ridge_regression":
        w, _ = ridge_regression(y_train, x_train, lambda_)
    else:
        w, _ = least_squares(y_train, x_train)

    # predict the test fold data
    y_pred = x_test @ w
    y_pred = classifictaion_pred(y_pred, using0_class=True, model_name=f)

    # Compute testing loss, F1 score, and accuracy
    loss_te = compute_neg_log_likelihood_loss(y_test, x_test, w)
    f1_score = compute_f1_score(y_test, y_pred)
    acc = compute_acc(y_test, y_pred)

    return loss_te, f1_score, acc


def cross_val(tx, y, max_iters, gammas, lambdas, regulizer_orders, f):
    """
    Choose the best hyperparameters for a model using cross-validation.

    Parameters:
    - tx (numpy.ndarray): Feature matrix of shape (N, D).
    - y (numpy.ndarray): Label vector of shape (N,).
    - max_iters (list): List of maximum iteration values to search through.
    - gammas (list): List of learning rate values to search through.
    - lambdas (list): List of lambda (regularization strength) values to search through.
    - regulizer_orders (list): List of regularization orders to search through.
    - f (function): A function for logistic regression or other related tasks.

    Returns:
    - best_lambda (float): The best lambda value found during cross-validation.
    - best_max_iters (int): The best maximum iteration value found during cross-validation.
    - best_gamma (float): The best gamma (learning rate) found during cross-validation.
    - best_reg (int): The best regularization order found during cross-validation.
    """
    # initilize w
    initial_w = np.zeros(tx.shape[1])

    # set the seed and the k_fold
    seed = 12
    k_fold = 10

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # Set the best hyper parameters names depending on the function name argument
    best_hyper_param_names = [
        "negative log liklihood loss",
        "f1_score",
        "accuracy",
        "lambda",
        "max iteration",
        "gamma",
        "regulizer",
    ]

    # Define lists to store the results of cross-validation
    cross_val_params = []

    # perform cross validation for all possible combinations of hyper parameters
    for lambda_ in lambdas:
        for max_iter in max_iters:
            for gamma in gammas:
                for reg in regulizer_orders:
                    mean_cross_val_loss, cross_val_loss = cross_validation_helper(
                        y,
                        tx,
                        k_indices,
                        f,
                        initial_w,
                        max_iter,
                        gamma,
                        lambda_,
                        reg,
                        k_fold,
                    )

                    print(
                        f" lambda = {lambda_}, iterations = {max_iter}, gamma = {gamma} using L{reg} : F1 = {mean_cross_val_loss[1]:.3f} and acc = {mean_cross_val_loss[2]:.3f}"
                    )
                    cross_val_params.append(
                        append_cross_val_result(
                            mean_cross_val_loss,
                            cross_val_loss,
                            lambda_,
                            max_iter,
                            gamma,
                            reg,
                        )
                    )

    # save all parameters and results for further analyis
    with open(f"model_performance/{f}_performance.pkl", "wb") as file:
        pickle.dump(cross_val_params, file)

    # transform the cross validation parameter list into a numpy array
    cross_val_params = np.array([x[:-1] for x in cross_val_params])

    # select best model based on f1 score
    best_idx = compute_best_hyper_params(cross_val_params, f, best_hyper_param_names)

    # cast to int max iterations and regularization order
    best_lambda = cross_val_params[best_idx][3]
    best_max_iters = int(cross_val_params[best_idx][4])
    best_gamma = cross_val_params[best_idx][5]
    best_reg = int(cross_val_params[best_idx][6])

    return (
        cross_val_params[best_idx][1],
        cross_val_params[best_idx][2],
        best_lambda,
        best_max_iters,
        best_gamma,
        best_reg,
    )


def compute_best_hyper_params(cross_val_params, f, best_hyper_param_names):
    """
    Computes the best hyperparameters based on cross-validation results and prints them.

    Parameters:
    cross_val_params (numpy.ndarray): A 2D array containing cross-validation results.
        Each row represents a set of hyperparameters, and the second column is F1 score.
    f (str): The name of the model.
    best_hyper_param_names (list): A list of strings representing the names of hyperparameters.

    Returns:
    int: The index of the row with the best hyperparameters based on the specified evaluation metric.

    """

    print("-----------------------------------------")
    print(f"Cross validation result for {f}")
    # select best model based on f1 score
    best_idx = np.argmax(cross_val_params[:, 1])

    # print the best hyper parameters
    for i in range(len(best_hyper_param_names)):
        print(f"best {best_hyper_param_names[i]} = {cross_val_params[best_idx][i]}")

    print("-----------------------------------------")
    return best_idx


def cross_validation_helper(
    y, tx, k_indices, f, initial_w, max_iter, gamma, lambda_, reg, k_fold
):
    """
    Perform k-fold cross-validation with given parameters and evaluation function.

    Parameters:
    y (numpy.ndarray): The target variable.
    tx (numpy.ndarray): The feature matrix.
    k_indices (numpy.ndarray): An array containing k-fold indices for data partitioning.
    f (str): the function name
    initial_w (numpy.ndarray): Initial model weights.
    max_iter (int): The maximum number of iterations for optimization algorithms.
    gamma (float): The learning rate for gradient descent or other optimization methods.
    lambda_ (float): Regularization parameter.
    reg (str): Regularization method ('L2' or 'L1').
    k_fold (int): The number of folds for cross-validation.

    Returns:
    tuple: A tuple containing mean cross-validation loss (or evaluation metric) and an array of cross-validation results.

    """
    cross_val_loss = np.array(
        [
            cross_validation_step(
                y,
                tx,
                k_indices,
                k,
                f,
                initial_w,
                max_iter,
                gamma,
                lambda_,
                order=reg,
            )
            for k in range(k_fold)
        ]
    )

    # compute the mean of the f1 score, the accuray and the test loss
    mean_cross_val_loss = cross_val_loss.mean(axis=0)

    return mean_cross_val_loss, cross_val_loss


def append_cross_val_result(
    mean_cross_val_loss, cross_val_loss, lambda_, max_iter, gamma, reg
):
    """
    Append cross-validation results and parameters to a tuple for logging or storage.

    Parameters:
    mean_cross_val_loss (numpy.ndarray): An array containing mean cross-validation losses or evaluation metrics.
    cross_val_loss (numpy.ndarray): An array of individual cross-validation results.
    lambda_ (float): Regularization parameter used in the cross-validation.
    max_iter (int): Maximum number of iterations used in optimization.
    gamma (float): Learning rate used in optimization.
    reg (str): Regularization method ('L2' or 'L1') used in optimization.

    Returns:
    tuple: A tuple containing mean cross-validation loss or evaluation metric components, along with relevant parameters
           and the array of individual cross-validation results.

    """
    return (
        mean_cross_val_loss[0],  # negative log likelihood loss
        mean_cross_val_loss[1],  # f1 score
        mean_cross_val_loss[2],  # accuracy
        lambda_,
        max_iter,
        gamma,
        reg,
        cross_val_loss,
    )
