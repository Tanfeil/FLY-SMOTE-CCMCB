"""
Module for loading and processing data for hotel fault prediction.

This module handles the loading of `.npy` files, each representing sensor data from different hotels.
It combines the data into a single DataFrame, assigns appropriate hotel IDs, and generates datasets
with specified positive-to-negative label ratios for model training purposes.

The module includes functionality to:
- Load `.npy` files and combine the data into a unified DataFrame with metadata.
- Create datasets with a given ratio of positive (fault) and negative (no fault) samples.
- Save the generated datasets as CSV files for further use.
"""

import numpy as np
import pandas as pd

# List of .npy files and corresponding hotel IDs
npy_files = ["F1_Equal.npy", "F2_Equal.npy", "F3_Equal.npy", "F4_Equal.npy"]
hotel_values = [0, 1, 2, 3]

# Feature and column names for DataFrame
# TODO: Which column name is the one too much?
column_names = [
    "evaporator_inlet_water_temperature",
    "evaporator_outlet_water_temperature",
    "condenser_inlet_water_temperature",
    "condenser_outlet_water_temperature",
    "evaporator_cooling_capacity",
    "compressor_inlet_air_temperature",
    "compressor_outlet_air_temperature",
    "evaporator_inlet_air_pressure",
    "condenser_outlet_air_pressure",
    "exhaust_air_overheat_temperature",
    "main_circuit_coolant_level",
    "main_coolant_pipe_valve_opening_size",
    "compressor_load",
    "compressor_current",
    "compressor_rotational_speed",
    "compressor_voltage",
    "compressor_power",
    "compressor_inverter_temperature",
    "fault"
]

# Short feature names for ease of use
feature_names = [
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "fault"
]


# Load and combine data from all .npy files into a single DataFrame
def load_and_combine_data(npy_files, hotel_values, feature_names):
    """
    Load .npy files, assign hotel IDs, and combine them into a single DataFrame.

    Args:
        npy_files (list): List of .npy file paths.
        hotel_values (list): Corresponding hotel IDs for the .npy files.
        feature_names (list): Column names for the DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with all data.
    """
    data_frames = []
    for file, hotel_id in zip(npy_files, hotel_values):
        # Load data and create a DataFrame
        data = np.load(f"./raw/{file}")
        df = pd.DataFrame(data, columns=feature_names)

        # Add metadata columns
        df['hotel'] = hotel_id
        df['fault'] = df['fault'].astype(int)

        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)


def create_datasets(positive_data, negative_data, ratios):
    """
    Create datasets with specified positive-to-negative label ratios.

    Args:
        positive_data (pd.DataFrame): DataFrame containing positive labels.
        negative_data (pd.DataFrame): DataFrame containing negative labels.
        ratios (list): List of desired positive-to-negative label ratios.

    Returns:
        dict: Dictionary of datasets for each ratio.
    """
    datasets = {}
    for ratio in ratios:
        # Calculate the number of positive samples needed
        num_positive_samples = int(len(negative_data) / ratio)

        # Randomly sample the required number of positives
        sampled_positive_data = positive_data.sample(num_positive_samples, random_state=42)

        # Combine positives and negatives
        combined_data = pd.concat([negative_data, sampled_positive_data], ignore_index=True)

        # Shuffle the combined dataset
        shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

        datasets[ratio] = shuffled_data

    return datasets


def generate_and_save_datasets(data, ratios, output_path):
    """
    Generate datasets with given ratios and save them as CSV files.

    Args:
        data (pd.DataFrame): Full dataset containing all samples.
        ratios (list): List of positive-to-negative label ratios.
        output_path (str): Path to save the datasets.
    """
    positive_data = data[data['fault'] == 1]
    negative_data = data[data['fault'] == 0]

    # Create datasets for each ratio
    datasets = create_datasets(positive_data, negative_data, ratios)

    # Save each dataset as a separate CSV file
    for ratio, dataset in datasets.items():
        dataset.to_csv(f"{output_path}-ratio1-{ratio}.csv", index=False)


if __name__ == "__main__":
    # Load and combine all data
    combined_data = load_and_combine_data(npy_files, hotel_values, feature_names)

    # Save the combined data to a CSV file
    combined_data.to_csv("hotels.csv", index=False)

    # Ratios for positive-to-negative samples
    imbalance_ratios = [4, 10, 20, 30]

    # Generate datasets with specified ratios
    generate_and_save_datasets(combined_data, imbalance_ratios, "./imbalance/hotels")

    # Generate separate datasets for each hotel with a fixed ratio of 1:4
    for hotel in hotel_values:
        hotel_data = combined_data[combined_data['hotel'] == hotel]
        generate_and_save_datasets(hotel_data, [4], f"./imbalance_separate_1-4/hotel{hotel}")
