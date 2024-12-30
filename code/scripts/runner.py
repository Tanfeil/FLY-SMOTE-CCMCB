import subprocess
import json
from itertools import product
import numpy as np


def load_params(param_file):
    """Lädt die Parameter aus der JSON-Datei und erzeugt alle Kombinationen."""
    with open(param_file, "r") as f:
        raw_params = json.load(f)

    full_param_list = []

    for param_set in raw_params:
        resolved_params = {}

        # Iteriere durch die Schlüssel und löse Ranges auf
        for key, value in param_set.items():
            if isinstance(value, dict) and "range" in value:
                # Konvertiere Range zu einer Liste
                start, stop, step = value["range"]
                resolved_params[key] = list(np.arange(start, stop + step, step))
            elif isinstance(value, list):
                # Behalte Listen so wie sie sind
                resolved_params[key] = value
            else:
                # Einzelne Werte in eine Liste verpacken
                resolved_params[key] = [value]

        # Erzeuge alle Kombinationen der Parameter

        keys, values = zip(*resolved_params.items())
        for combination in product(*values):
            full_param_list.append(dict(zip(keys, combination)))

    return full_param_list


def batch_run(param_file="./config/params.json"):
    params_list = load_params(param_file)

    for i, params in enumerate(params_list):
        print(f"Running variant {i + 1} with parameters: {params}")

        # Erstelle den Befehl mit den Parametern
        cmd = ["python", "python", "-m", "code.FLY-SMOTE-CCMCB.main"]  # Pfad zu `main.py`
        for key, value in params.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))

        if params["wandb_logging"] and "wandb_name" not in params:
            cmd.append("--wandb_name")
            cmd.append(f"{params['dataset_name']}_FLY-SMOTE_k{params['k_value']}_r{params['r_value']}_t{params['threshold']}")

        cmd.append("--seed")
        cmd.append(str(42))

        # Starte das Skript als Subprozess
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Ausgabe des Ergebnisses
        print(f"Variant {i + 1} finished with return code {result.returncode}")
        print(f"Standard Output:\n{result.stdout}")
        print(f"Standard Error:\n{result.stderr}")


if __name__ == "__main__":
    batch_run()
