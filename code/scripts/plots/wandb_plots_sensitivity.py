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
    "sensitivity-k": "k_value",
    "sensitivity-r": "r_value",
    "sensitivity-g-gan": "g_value",
    "sensitivity-k-gan": "k_gan_value",
    "sensitivity-r-gan": "r_gan_value",

}

# List of datasets and their corresponding split types
datasets = ["adult"]

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
            history["project"] = project
            history["dataset"] = dataset
            m = method(run.config)

            # Assign a unique identifier for the run
            history["run_id"] = f"{m}"
            history[projects[project]] = run.config[projects[project]]

            history_step_39 = history[history["round"] == 40]
            history_step_39 = history_step_39[history_step_39["sensitivity"] >= 0]

            results.append(history_step_39)


# Aggregate mean and standard deviation of balanced accuracy by group
df = pd.concat(results)
grouped = df.reset_index()

# Set Seaborn style for consistent visualization
sns.set(style="whitegrid")

# Generate and save plots for each project and dataset
for project in projects.keys():

    project_data = grouped[grouped["project"] == project]
    for dataset in datasets:
        plt.figure(figsize=(12, 6))
        dataset_data = project_data[project_data["dataset"] == dataset]


        # Create line plot for mean balanced accuracy over rounds
        sns.lineplot(
            data=dataset_data,
            x=projects[project],
            y="sensitivity",
            markers=True
        )

        # Adjust plot aesthetics
        plt.ylabel("Sensitivity", fontsize=16)
        plt.xlabel(projects[project], fontsize=16)
        plt.xticks(dataset_data[projects[project]], fontsize=14)  # X-Ticks
        plt.yticks(fontsize=14)  # Y-Ticks
        plt.ylim(0, 1)  # Scale
        #plt.legend(fontsize=16)
        plt.tight_layout()

        # Save plot to file
        output_path = os.path.join(output_dir, f"{dataset}_{project}.png")
        plt.savefig(output_path)

        #plt.show()
        plt.close()