import subprocess
import json
from itertools import product
import numpy as np
from concurrent.futures import ProcessPoolExecutor


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


def run_variant(params):
    """Führt eine einzelne Variante aus."""
    print(f"Running with parameters: {params}")

    # Erstelle den Befehl mit den Parametern
    cmd = ["python", "code/FLY-SMOTE-CCMCB_parallel/main.py"]  # Pfad zu `main.py`
    for key, value in params.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    if params.get("wandb_logging") and "wandb_name" not in params:
        cmd.append("--wandb_name")
        cmd.append(f"{params['dataset_name']}_FLY-SMOTE_k{params['k_value']}_r{params['r_value']}_t{params['threshold']}")

    cmd.append("--seed")
    cmd.append(str(42))

    # Starte das Skript als Subprozess
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Rückgabe des Ergebnisses
    return {
        "params": params,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def batch_run(param_file="./config/params.json", max_workers=4):
    """Führt alle Varianten parallel aus."""
    params_list = load_params(param_file)

    # Nutze einen ProcessPoolExecutor für parallele Ausführung
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_variant, params_list))

    # Verarbeite und drucke die Ergebnisse
    for i, result in enumerate(results):
        print(f"Variant {i + 1} finished with return code {result['returncode']}")
        print(f"Standard Output:\n{result['stdout']}")
        print(f"Standard Error:\n{result['stderr']}")


if __name__ == "__main__":
    batch_run()
