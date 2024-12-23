import numpy as np

from code.shared.ReadData import ReadData

def read_data(file_name, directory):
    data_loader = ReadData(file_name)
    return data_loader.load_data(directory)


def check_imbalance(y_data):
    counts = np.bincount(y_data)
    num_zeros, num_ones = counts[0], counts[1]

    if num_zeros < num_ones:
        minority_label = 0
        threshold = num_zeros / num_ones
    else:
        minority_label = 1
        threshold = num_ones / num_zeros

    return minority_label, threshold
