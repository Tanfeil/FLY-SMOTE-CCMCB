import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

wandb.login()
api = wandb.Api()

output_dir = "/mnt/c/Users/jonat/OneDrive/Dokumente/Uni/III/FL/Template_FLCourse_Report/images/"
os.makedirs(output_dir, exist_ok=True)

projects = {
    "multiple-wo-seed": ["FdgAvg", "FLY-SMOTE", "FLY-SMOTE-CCMCB"],
}

datasets = ["bank"]
splits = {None: "random", 0: "age"}


def method(config):
    if config["ccmcb"]:
        return "FLY-SMOTE-CCMCB"
    if not config["ccmcb"] and config["threshold"] > 0:
        return "FLY-SMOTE"
    return "FdgAvg"


results = []

for project, methods in projects.items():
    for dataset in datasets:
        project_name = f"{dataset}_{project}"
        runs = api.runs(f"Tanfeil/{project_name}")

        for run in runs:
            history = run.history()
            history["project"] = project
            history["dataset"] = dataset
            m = method(run.config)
            spl = splits[run.config["attribute_index"]]

            history["run"] = run.id
            history["run_id"] = f"{m}_{spl}"

            results.append(history)

df = pd.concat(results).reset_index()
grouped = df

sns.set(style="whitegrid")
for project in projects.keys():

    project_data = grouped[grouped["project"] == project]
    for dataset in datasets:
        plt.figure(figsize=(12, 6))
        dataset_data = project_data[project_data["dataset"] == dataset]

        sns.lineplot(
            data=dataset_data,
            x="round",
            y="balanced_accuracy",
            hue="run_id",
            markers=True,
            palette="muted",
            errorbar=("ci", 95)
        )

        # Plot-Anpassungen
        plt.ylabel("Mean Balanced Accuracy", fontsize=16)
        plt.xlabel("Training Round", fontsize=16)
        plt.xticks(fontsize=14)  # X-Ticks
        plt.yticks(fontsize=14)  # Y-Ticks
        plt.ylim(0.5, 0.85)  # Scale
        plt.legend(fontsize=16)

        # Plot anzeigen oder speichern
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{dataset}_{project}_with-std.png")
        plt.savefig(output_path)
        # plt.show()
        plt.close()
