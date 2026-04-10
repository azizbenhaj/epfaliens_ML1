"""Some helper functions for project 1."""
import csv
import numpy as np
import os


def load_csv_data(data_path):
    """
    This function loads the data and returns the respectinve numpy arrays.

    Args:
        data_path (str): datafolder path

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    print("start loading the data")

    # Load y_train data
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )

    # Load x_train data
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"),
        delimiter=",",
        skip_header=1,
    )

    # Load x_test data
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"),
        delimiter=",",
        skip_header=1,
    )

    # get the sample indices
    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)

    # ignore the indices column
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    print("finish loading the data")

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """

    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
