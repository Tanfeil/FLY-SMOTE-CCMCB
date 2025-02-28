"""
Module for downloading and extracting datasets based on configuration.

This module handles the download of dataset files from specified URLs, extraction of ZIP files,
and recursive extraction of nested ZIP files. It loads dataset configurations from a JSON file and
manages the storage of datasets in a specified directory.
"""

import json
import os
from zipfile import ZipFile

import requests


def get_base_dir():
    """
    Returns the base directory of the script.

    This function computes the absolute path of the base directory by traversing the directory structure
    upwards from the current script.

    Returns:
        str: The absolute path to the base directory.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_path(*path_segments):
    """
    Constructs a path relative to the base directory.

    This function combines the base directory with additional path components to generate an absolute
    path relative to the project's root directory.

    Args:
        path_segments (str): Components of the relative path.

    Returns:
        str: The absolute path.
    """
    return os.path.join(get_base_dir(), *path_segments)


def download_file(url, save_path):
    """
    Downloads a file from the specified URL and saves it to the given path.

    This function uses the requests library to download the content from the provided URL and saves it
    as a file on the local system.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file will be saved.

    Raises:
        Exception: If the download fails for any reason, an exception is raised.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def extract_zip(file_path, extract_to):
    """
    Recursively extracts a ZIP file and handles nested ZIPs.

    This function extracts the content of a ZIP file to the specified directory. If any nested ZIP
    files are found during extraction, they are also extracted recursively.

    Args:
        file_path (str): The path to the ZIP file to extract.
        extract_to (str): The directory to extract the contents to.

    Raises:
        Exception: If the extraction fails, an exception is raised.
    """
    try:
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted: {file_path} to {extract_to}")

        # Now look for any ZIP files inside the extracted folder and extract them
        for root, _, files in os.walk(extract_to):
            for file in files:
                if file.endswith(".zip"):
                    zip_file_path = os.path.join(root, file)
                    new_extract_to = os.path.splitext(zip_file_path)[0]
                    os.makedirs(new_extract_to, exist_ok=True)
                    print(f"Found nested ZIP: {zip_file_path}, extracting to {new_extract_to}")
                    extract_zip(zip_file_path, new_extract_to)  # Recursive extraction
                    os.remove(zip_file_path)  # Optionally remove nested ZIP after extraction

    except Exception as e:
        print(f"Failed to extract {file_path}: {e}")


def load_config(config_path):
    """
    Loads the dataset configuration from a JSON file.

    This function loads a JSON file containing the dataset configuration, which includes URLs and
    target names for each dataset to be downloaded.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        list: List of dataset configurations, where each configuration is a dictionary containing
              the URL and target name for a dataset.

    Raises:
        Exception: If the configuration file cannot be read or is invalid, an exception is raised.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load configuration file {config_path}: {e}")
        return []


def main():
    """
    Downloads and extracts the datasets into the `dataset` folder based on the configuration.

    This function orchestrates the process of downloading and extracting datasets as specified in
    the dataset configuration file. It ensures the datasets are saved in the appropriate directories.

    Raises:
        Exception: If any step fails, an exception is raised.
    """
    # Define paths
    dataset_dir = get_path("dataset")
    config_path = get_path("config", "datasets_config.json")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Load dataset configurations
    datasets = load_config(config_path)

    for dataset in datasets:
        target_path = os.path.join(dataset_dir, dataset["target_name"])

        if not os.path.exists(target_path):
            download_file(dataset["url"], target_path)

            # If the file is a ZIP, extract it
            if target_path.endswith(".zip"):
                extract_to = os.path.splitext(target_path)[0]
                extract_zip(target_path, extract_to)
                # Optionally remove the ZIP file after extraction
                os.remove(target_path)
                print(f"Removed ZIP file: {target_path}")
        else:
            print(f"File already exists: {target_path}")


if __name__ == "__main__":
    main()
