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
        #TODO: should be never division by zero? except no zeros neither ones?
        threshold = num_zeros / (num_ones + 1e-13)
        len_minor = num_zeros
        len_major = num_ones
    else:
        minority_label = 1
        threshold = num_ones / (num_zeros + 1e-13)
        len_minor = num_ones
        len_major = num_zeros

    return minority_label, threshold, len_minor, len_major
