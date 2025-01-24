import numpy as np

from .ReadData import ReadData

def read_data(file_name, directory):
    data_loader = ReadData(file_name)
    return data_loader.load_data(directory)


def check_imbalance(y_data):
    assert len(y_data) > 0, "data should not be empty for balance check"

    counts = np.bincount(y_data)
    num_zeros = counts[0]
    if counts.size == 2:
        num_ones = counts[1]
    else:
        num_ones = 0

    if num_zeros < num_ones:
        minority_label = 0
        threshold = num_zeros / (num_ones + 1e-13)
    else:
        minority_label = 1
        threshold = num_ones / (num_zeros + 1e-13)

    return minority_label, threshold
