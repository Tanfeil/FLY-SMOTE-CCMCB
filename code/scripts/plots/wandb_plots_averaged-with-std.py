# SPDX-FileCopyrightText: 2025 Jonathan Feilmeier
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for generating and saving plots of the mean balanced accuracy for different projects and datasets.

This module interacts with Weights & Biases (W&B) to fetch the results of experiments for multiple projects,
processes the data to calculate the mean balanced accuracy for different runs, and generates line plots showing
the results over training rounds. It also visualizes the results with confidence intervals.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

# Authenticate to Weights & Biases
wandb.login()
api = wandb.Api()

# Output directory for saving plots
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# Define projects and their associated methods
projects = {
    "multiple-wo-seed": ["FdgAvg", "FLY-SMOTE", "FLY-SMOTE-CCMCB"],
}

# List of datasets and their corresponding split types
datasets = ["bank"]
splits = {None: "random", 0: "age"}


def method(config):
    """
    Determine the method type based on the configuration parameters.

    Args:
        config (dict): Configuration dictionary of a W&B run containing various hyperparameters and settings.

    Returns:
        str: The method type determined by the configuration (e.g., "FLY-SMOTE-CCMCB").
    """
    if config["ccmcb"]:
        return "FLY-SMOTE-CCMCB"
    if not config["ccmcb"] and config["threshold"] > 0:
        return "FLY-SMOTE"
    return "FdgAvg"


# List to collect all historical results
results = []

# Process runs for each project and dataset
for project, methods in projects.items():
    for dataset in datasets:
        project_name = f"{dataset}_{project}"
        runs = api.runs(f"Tanfeil/{project_name}")

        for run in runs:
            # Retrieve run history and add metadata
            history = run.history()
            history["project"] = project
            history["dataset"] = dataset
            m = method(run.config)
            spl = splits[run.config["attribute_index"]]

            # Assign a unique identifier for the run
            history["run"] = run.id
            history["run_id"] = f"{m}_{spl}"

            results.append(history)

# Concatenate all results into a single DataFrame and reset the index
df = pd.concat(results).reset_index()
grouped = df

# Set Seaborn style for consistent visualization
sns.set(style="whitegrid")

# Generate and save plots for each project and dataset
for project in projects.keys():
    project_data = grouped[grouped["project"] == project]
    for dataset in datasets:
        plt.figure(figsize=(12, 6))
        dataset_data = project_data[project_data["dataset"] == dataset]

        # Create line plot for mean balanced accuracy over rounds with confidence intervals
        sns.lineplot(
            data=dataset_data,
            x="round",
            y="balanced_accuracy",
            hue="run_id",
            markers=True,
            palette="muted",
            errorbar=("ci", 95)
        )

        # Adjust plot aesthetics
        plt.ylabel("Mean Balanced Accuracy", fontsize=16)
        plt.xlabel("Training Round", fontsize=16)
        plt.xticks(fontsize=14)  # X-Ticks
        plt.yticks(fontsize=14)  # Y-Ticks
        plt.ylim(0.5, 0.85)  # Scale
        plt.legend(fontsize=16)
        plt.tight_layout()

        # Save plot to file
        output_path = os.path.join(output_dir, f"{dataset}_{project}.png")
        plt.savefig(output_path)

        # plt.show()
        plt.close()
