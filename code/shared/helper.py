import numpy as np
from .DatasetLoader import DatasetLoader


def load_data_from_file(file_name, directory):
    """
    Loads data from the specified file in the given directory.

    Args:
        file_name (str): The name of the file to load.
        directory (str): The directory where the file is located.

    Returns:
        Data loaded from the file.
    """
    data_loader = DatasetLoader(file_name)
    return data_loader.load_data(directory)


def check_class_imbalance(labels):
    """
    Checks if the dataset is imbalanced and returns the imbalance details.

    Args:
        labels (np.ndarray): Array containing the labels of the dataset.

    Returns:
        tuple:
            - minority_label (int): The label of the minority class.
            - imbalance_threshold (float): The imbalance ratio of the minority class to the majority class.
            - minority_class_size (int): The number of samples in the minority class.
            - majority_class_size (int): The number of samples in the majority class.
    """
    if len(labels) == 0:
        raise ValueError("The dataset should not be empty for balance check")

    # Count the number of occurrences of each label
    label_counts = np.bincount(labels)
    num_minority_class = label_counts.min()  # Either the zeros or the ones
    num_majority_class = label_counts.max()  # The other class

    # Determine the minority and majority labels based on counts
    if num_minority_class == label_counts[0]:
        minority_label = 0
    else:
        minority_label = 1

    assert num_majority_class != 0, "majority class should not be Zero"

    imbalance_threshold = num_minority_class / num_majority_class

    return minority_label, imbalance_threshold, num_minority_class, num_majority_class