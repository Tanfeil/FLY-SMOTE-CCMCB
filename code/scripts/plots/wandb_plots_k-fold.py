import os

import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Authenticate to Weights & Biases
wandb.login()
api = wandb.Api()

# Output directory for saving plots
output_dir = "/mnt/c/Users/jonat/OneDrive/Dokumente/Uni/III/FL/Template_FLCourse_Report/images/"
os.makedirs(output_dir, exist_ok=True)

# Define projects and their associated methods
projects = {
    "k-fold": ["FLY-SMOTE-CCMCB"],
}

# List of datasets and their corresponding split types
datasets = ["bank"]
splits = {None: "random", 0: "age"}

def method(config):
    """
    Determine the method type based on the configuration parameters.
    Args:
        config (dict): Configuration dictionary of a W&B run.
    Returns:
        str: The method type (e.g., "FLY-SMOTE-CCMCB").
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
            history.reset_index(drop=True, inplace=True)  # Indizes zurücksetzen
            history["project"] = project
            history["dataset"] = dataset
            history["method"] = method(run.config)
            history["split"] = splits[run.config["attribute_index"]]

            # Assign a unique identifier for the run
            history["run_id"] = run.name.replace("None_", "")  # Eindeutige Run-Kennung hinzufügen

            if "balanced_accuracy" in history.columns:
                results.append(history)

df = pd.concat(results, ignore_index=True)

if "round" not in df.columns or "balanced_accuracy" not in df.columns:
    raise ValueError("Die Spalten `round` und/oder `balanced_acc` fehlen in den Daten.")

# Set Seaborn style for consistent visualization
sns.set(style="whitegrid")

# Generate and save plots for each project, dataset and spli
for project, methods in projects.items():
    for m in  methods:
        for dataset in datasets:
            for split in splits.values():
                plt.figure(figsize=(12, 6))

                # filter data
                project_data = df[
                    (df["project"] == project) &
                    (df["split"] == split) &
                    (df["dataset"] == dataset) &
                    (df["method"] == m)
                    ]

                # Create line plot for mean balanced accuracy over rounds
                sns.lineplot(
                    data=project_data,
                    x="round",
                    y="balanced_accuracy",
                    hue="run_id",  # Jeder Run erhält eine eigene Farbe
                    alpha=0.8,
                    legend=True
                )

                # Adjust plot aesthetics
                plt.ylabel("Balanced Accuracy", fontsize=16)
                plt.xlabel("Training Round", fontsize=16)
                plt.xticks(fontsize=14)  # X-Ticks
                plt.yticks(fontsize=14)  # Y-Ticks
                plt.ylim(0.5, 0.85)  # Skalierung
                plt.legend(fontsize=16)
                plt.tight_layout()

                # Save plot to file
                output_path = os.path.join(output_dir, f"{dataset}_{project}_{m}_{split}.png")
                plt.savefig(output_path)

                # plt.show()
                plt.close()
