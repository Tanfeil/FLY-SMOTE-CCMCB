import os

import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wandb.login()
api = wandb.Api()

output_dir = "/mnt/c/Users/jonat/OneDrive/Dokumente/Uni/III/FL/Template_FLCourse_Report/images/"
os.makedirs(output_dir, exist_ok=True)

projects = {
    #"k-fold": ["FLY-SMOTE-CCMCB"],
    "multiple-v1": ["FdgAvg", "FLY-SMOTE", "FLY-SMOTE-CCMCB"]
    #"woseed_multiple": ["FdgAvg", "FLY-SMOTE", "FLY-SMOTE-CCMCB"],
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

                    history["run_id"] = f"{m}_{spl}"

                    results.append(history)

df = pd.concat(results)
grouped = df.groupby(["round", "project", "dataset", "run_id"]).agg(
    mean_balanced_acc=('balanced_accuracy', 'mean'),
    std_balanced_acc=('balanced_accuracy', 'std'),
).reset_index()

sns.set(style="whitegrid")
for project in projects.keys():
    plt.figure(figsize=(12, 6))

    project_data = grouped[grouped["project"] == project]
    for dataset in datasets:
        project_data = project_data[project_data["dataset"] == dataset]

        sns.lineplot(
            data=project_data,
            x="round",
            y="mean_balanced_acc",
            hue="run_id",
            markers=True,
            palette="muted"
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

        output_path = os.path.join(output_dir, f"{dataset}_{project}.png")
        plt.savefig(output_path)
        #plt.show()