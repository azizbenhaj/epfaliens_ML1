import numpy as np
from implementations import *
import time
from helpers import load_csv_data, create_csv_submission
from preprocess_data import preprocess, add_bias, pca
from cross_validation import cross_val

DATA_FOLDER = "data/dataset_to_release/"


def model(tx, y, max_iters, gammas, lambdas, regulizer_orders):
    """
    Train a regularized logistic regression model using cross-validation to find the best hyperparameters.
    Train the model with the best hyper parameters on the given dataset

    Parameters:
    tx (numpy.ndarray): Feature matrix of shape (num_samples, num_features).
    y (numpy.ndarray): Label vector of shape (num_samples,).
    max_iters (int): Maximum number of iterations for the logistic regression optimization.
    gammas (list): List of learning rate values to search through during cross-validation.
    lambdas (list): List of lambda (regularization strength) values to search through during cross-validation.
    regulizer_orders (list): List of regularization orders to search through during cross-validation.

    Returns:
    w (numpy.ndarray): Weight vector of the trained logistic regression model.
    loss (float): Loss value of the trained model.
    """

    initial_w = np.zeros(tx.shape[1])
    functions = [
        "logistic_regression + / - regulizer",
        "using ridge / lasso / linear regression",
    ]
    best_hyper_params = []

    # Perform cross-validation to find the best hyperparameters for the ridge regression model
    print("using ridge / lasso / linear regression")
    best_hyper_params.append(
        cross_val(
            tx,
            y,
            [0],
            [0],
            lambdas,
            regulizer_orders,
            "ridge_regression",
        )
    )

    # Perform cross-validation to find the best hyperparameters for the logistic_regression model
    print("using logistic_regression + / - regulizer")
    best_hyper_params.append(
        cross_val(
            tx,
            y,
            max_iters,
            gammas,
            lambdas,
            regulizer_orders,
            "logistic_regression",
        )
    )

    best_hyper_params = np.array(best_hyper_params)

    # Determine the best model based on a combination of F1 score and accuracy
    best_idx = np.argmax(
        0.75 * best_hyper_params[:, 0] + 0.25 * best_hyper_params[:, 1] / 100
    )
    best_lambda = best_hyper_params[best_idx][2]
    best_max_iters = int(best_hyper_params[best_idx][3])
    best_gamma = best_hyper_params[best_idx][4]
    best_reg = int(best_hyper_params[best_idx][5])
    best_model = functions[best_idx]

    # Print the cross-validation results
    print_cross_validation_results(best_hyper_params, functions)

    w = 0
    # Train the selected best model
    if best_model == "logistic_regression":
        w, _ = reg_logistic_regression_adaptive_lr(
            y, tx, best_lambda, initial_w, best_max_iters, best_gamma, order=best_reg
        )
    else:
        w, _ = ridge_regression(y, tx, best_lambda)

    return w, best_model


if __name__ == "__main__":
    start = time.time()

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path=DATA_FOLDER)

    # Preprocess the data
    x_train, x_test, y_train = preprocess(x_train, x_test, y_train)
    x_train, x_test = add_bias(x_train, x_test)

    # Perform PCA (Principal Component Analysis) on the data
    # projected_x_train, projected_x_test, eigvalues_sum = pca(x_train, x_test, num_components=50)
    projected_x_train, projected_x_test = x_train, x_test

    # Define hyperparameter search spaces
    print("start trainig")
    lambdas = [0, 0.1]
    max_iters = [200, 400]
    gammas = [0.1, 0.3]
    regulizer_orders = [1, 2]

    # Train the model and find the best hyperparameters
    w, best_model = model(
        projected_x_train, y_train, max_iters, gammas, lambdas, regulizer_orders
    )

    # Save the trained model parameters
    print("the best model is ", best_model)
    np.save("model_performance/model_parameters.npy", w)

    # Perform classification on the predictions and save the results
    y_pred = projected_x_test @ w
    y_pred = classifictaion_pred(y_pred, using0_class=False, model_name=best_model)
    create_csv_submission(test_ids, y_pred, name="submission1.csv")

    print(f"time needed = {time.time()-start} ")
