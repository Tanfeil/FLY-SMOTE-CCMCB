import subprocess
import json


def load_params(param_file):
    """Lädt die Parameter für die verschiedenen Varianten aus einer JSON-Datei."""
    with open(param_file, "r") as f:
        return json.load(f)


def batch_run(param_file="../../config/params.json"):
    params_list = load_params(param_file)

    for i, params in enumerate(params_list):
        print(f"Running variant {i + 1} with parameters: {params}")

        # Erstelle den Befehl mit den Parametern
        cmd = ["python", "../main.py"]  # Pfad zu `main.py`
        for key, value in params.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))

        if params["wandb_logging"]:
            cmd.append("--wandb_name")
            cmd.append(f"FLY-SMOTE_{params['dataset_name']}_k{params['k_value']}_r{params['r_value']}_t{params['threshold']}")

        # Starte das Skript als Subprozess
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Ausgabe des Ergebnisses
        print(f"Variant {i + 1} finished with return code {result.returncode}")
        print(f"Standard Output:\n{result.stdout}")
        print(f"Standard Error:\n{result.stderr}")


if __name__ == "__main__":
    batch_run()
