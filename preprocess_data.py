import numpy as np


def preprocess(x_train, x_test, y_train):
    """
    Preprocess the training and testing data for a machine learning task.

    Parameters:
    - x_train (numpy.ndarray): Training feature matrix.
    - x_test (numpy.ndarray): Testing feature matrix.
    - y_train (numpy.ndarray): Training target labels.

    Returns:
    - x_train (numpy.ndarray): Preprocessed training feature matrix.
    - x_test (numpy.ndarray): Preprocessed testing feature matrix.
    - y_train (numpy.ndarray): Preprocessed training target labels.
    """

    # remove columns with high percentage of nan
    x_train, x_test = remove_columns_with_high_nan(x_train, x_test)

    # replace nan values by the mean of the column
    x_train = clean_data(x_train)
    x_test = clean_data(x_test)

    # Remove column with a null std
    x_train, x_test = remove_null_std(x_train, x_test)

    # standardize values
    x_train, x_test = standardize(x_train, x_test)

    # outliers removal form the training set
    x_train, y_train = outliers_removal(x_train, y_train)

    # Replace -1 class by 0 to ensure proper compatibility with sigmoid and log operations
    y_train[y_train == -1] = 0

    # balance data
    x_train, y_train = balance_data(x_train, y_train)

    return x_train, x_test, y_train


def add_bias(x_train, x_test):
    """
    Add a bias (intercept) term to the feature matrices for training and testing data.

    Parameters:
    - x_train (numpy.ndarray): Training feature matrix.
    - x_test (numpy.ndarray): Testing feature matrix.

    Returns:
    - x_train (numpy.ndarray): Training feature matrix with a bias term added.
    - x_test (numpy.ndarray): Testing feature matrix with a bias term added.
    """
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    return x_train, x_test


def balance_data(x_train, y_train):
    """
    Balance the training data by downsampling the dominant class if there is a significant class imbalance.

    Parameters:
    - x_train (numpy.ndarray): Training feature matrix.
    - y_train (numpy.ndarray): Training target labels.

    Returns:
    - x_train (numpy.ndarray): Balanced training feature matrix.
    - y_train (numpy.ndarray): Balanced training target labels.
    """

    print("----------------------------------------------")

    # compute the percentage of each class
    class_0 = len(np.where(y_train == 0)[0])
    class_1 = len(np.where(y_train == 1)[0])
    pencentgae_class_0 = class_0 / y_train.shape[0]
    percentage_class_1 = class_1 / y_train.shape[0]
    print(f"class 0 = {class_0 :.2f} and class 1 = {class_1:.2f}")
    print(f"class 0 = {pencentgae_class_0 :.2f} and class 1 = {percentage_class_1:.2f}")

    # select among the dominant class the indices to remove
    if np.abs(pencentgae_class_0 - percentage_class_1) > 0.1:
        print("the data is unbalanced ")
        indices_to_drop = []
        if class_0 > class_1:
            print("downsampling class 0")
            indices_to_drop = downsample_data(
                np.where(y_train == 0)[0], class_0 - class_1 * 2
            )
        else:
            indices_to_drop = downsample_data(
                np.where(y_train == 1)[0], class_1 - class_0 * 2
            )

    # downsample the dominannt class
    x_train = np.delete(x_train, indices_to_drop, axis=0)
    y_train = np.delete(y_train, indices_to_drop, axis=0)

    # shuffle data
    random_indices = np.random.permutation(x_train.shape[0])
    x_train = x_train[random_indices]
    y_train = y_train[random_indices]

    print("data balanced")
    print("----------------------------------------------")
    return x_train, y_train


def downsample_data(indices, max_data):
    """
    Downsample data by randomly selecting a limited number of indices.

    Parameters:
    - indices (numpy.ndarray or list): Data indices to downsample from.
    - max_data (int): The maximum number of indices to retain.

    Returns:
    - downsampled_indices (numpy.ndarray or list): Randomly selected indices within the specified limit.
    """

    np.random.seed(42)
    indices = np.random.permutation(indices)
    return indices[:max_data]


def normalize(x_train, x_test):
    """Normalize the input data x_train and x_test using min max normalization

    Args:
        x: numpy array of shape=(num_samples, num_features)

    Returns:
        standartized data, shape=(num_samples, num_features)

    """
    # compute the mean, max and the range of each feature
    train_min = np.min(x_train, axis=0)
    train_max = np.max(x_train, axis=0)
    range_data = train_max - train_min
    range_data[range_data == 0] = 1

    # normalize the data
    x_train_normalized = (x_train - train_min) / range_data
    x_test_normalized = (x_test - train_min) / range_data

    print("standarising done")

    return x_train_normalized, x_test_normalized


def standardize(x_train, x_test):
    """Stadartize the input data x

    Args:
        x: numpy array of shape=(num_samples, num_features)

    Returns:
        standartized data, shape=(num_samples, num_features)

    """
    # compute the mean and the std of the training sample
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    # normalize the train and test dataset
    x_train_normalized = (x_train - train_mean) / train_std
    x_test_normalized = (x_test - train_mean) / train_std
    print("standarising done")
    return x_train_normalized, x_test_normalized


def outliers_removal(
    x_train_normalized, y_train, threshold=3, max_false_percentage=0.3
):
    """
    Remove outliers from the training data based on z-scores and a false percentage threshold.

    Parameters:
    - x_train_normalized (numpy.ndarray): Normalized training feature matrix.
    - y_train (numpy.ndarray): Training target labels.
    - threshold (float): Z-score threshold for defining outliers (default is 3).
    - max_false_percentage (float): Maximum acceptable percentage of False values in a row (default is 0.3).

    Returns:
    - x_train_cleaned (numpy.ndarray): Training feature matrix with outliers removed.
    - y_train_cleaned (numpy.ndarray): Training target labels with corresponding outliers removed.
    """

    z_scores = np.abs(x_train_normalized)

    # Create a mask of non-outliers
    outlier_mask = z_scores < threshold

    # Calculate the percentage of False values in each row of the mask
    false_percentages = 1 - np.mean(outlier_mask, axis=1)

    # Check if the false percentage is less than the threshold (10%)
    non_outlier_rows = false_percentages < max_false_percentage

    return x_train_normalized[non_outlier_rows], y_train[non_outlier_rows]


def clean_data(data):
    """
    Replace NaN values in x_train, x_test by the mean value of the column

    Args:
    - data (numpy.ndarray): Data with NaN values.

    Returns:
    - data (numpy.ndarray): Data without NaN values.

    """

    # Iterate through the columns
    for col_idx in range(data.shape[1]):
        column = data[:, col_idx]
        nan_indices = np.isnan(column)

        if np.any(nan_indices):
            # Calculate the mean of the column (excluding NaN values)
            column_mean = np.nanmean(column)

            # Replace NaN values with the calculated mean
            column[nan_indices] = column_mean

    return data


def remove_null_std(x_train, x_test):
    """
    Remove columns with a single value.

    Args:
    - x_train (numpy.ndarray) : Training data with columns with a single value
    - x_test (numpy.ndarray) : Testing data with columns with a single value

    Returns:
    - x_train (numpy.ndarray) : Training data without columns with a single value
    - x_test (numpy.ndarray) : Testing data without columns with a single value

    """
    # Calculate the standard deviation for each column (along axis 0)
    std_dev = np.std(x_train, axis=0)
    # Find columns with zero standard deviation
    zero_std_columns = np.where(std_dev <= 1e-8)[0]
    # Remove columns with zero standard deviation
    x_train = np.delete(x_train, zero_std_columns, axis=1)
    x_test = np.delete(x_test, zero_std_columns, axis=1)
    print("delete std ==0 ", x_train.shape)

    return x_train, x_test


def remove_columns_with_high_nan(x_train, x_test, threshold=0.01):
    """
    Remove columns with a NaN value percentage greater than or equal to the threshold.

    """

    # Calculate the NaN value percentage for each column
    nan_percentage = np.isnan(x_train).mean(axis=0)

    # Select columns where the NaN percentage is less than the threshold
    selected_columns = np.arange(x_train.shape[1])[nan_percentage < threshold]

    # Create a new array with the selected columns
    x_train = x_train[:, selected_columns]
    x_test = x_test[:, selected_columns]
    print("nan value removed ", x_train.shape)
    return x_train, x_test


def pca(x_train_standarized, x_test_standarized, info_loss=0):
    """
    Perform Principal Component Analysis (PCA) on standardized training and testing data.

    Parameters:
    - x_train_standardized (numpy.ndarray): Standardized training feature matrix.
    - x_test_standardized (numpy.ndarray): Standardized testing feature matrix.
    - info_loss: the maximum percentage of eigenvalues that we drop.

    Returns:
    - projected_x_train (numpy.ndarray): Training data projected onto the selected principal components.
    - projected_x_test (numpy.ndarray): Testing data projected onto the selected principal components.
    - info (float): Percentage of variance retained by the selected principal components.
    """

    #  Calculate the covariance matrix
    cov_mat = np.cov(x_train_standarized, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Calculate the total sum of eigenvalues
    total_eigenvalue_sum = np.sum(eigen_values)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Determine the number of components to retain based on info_loss
    retained_variance = 0
    retained_components = 0
    for i, eigenval in enumerate(sorted_eigenvalue):
        retained_variance += eigenval / total_eigenvalue_sum
        retained_components += 1
        if retained_variance >= 1.0 - info_loss:
            break

    # Select a subset of eigenvectors
    eigenvector_subset = sorted_eigenvectors[:, 0:retained_components]

    # Project data onto selected principal components
    projected_x_train = np.dot(eigenvector_subset.T, x_train_standarized.T).T
    projected_x_test = np.dot(eigenvector_subset.T, x_test_standarized.T).T

    return projected_x_train, projected_x_test
