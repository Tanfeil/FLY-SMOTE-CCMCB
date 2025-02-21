import logging
import subprocess
import json
from itertools import product
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import itertools
import argparse

from code.shared.logger_config import configure_logger

logger = logging.getLogger()

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
                start, stop, step = value["range"]
                resolved_params[key] = np.arange(start, stop + step, step).tolist()
            elif isinstance(value, dict) and "perm" in value:
                numbers = value["perm"]
                sets = []
                for r in range(1, len(numbers) + 1):
                    sets.extend(itertools.combinations(numbers, r))
                valid_sets = [sorted(s, reverse=value["reverse"]) for s in sets]
                resolved_params[key] = valid_sets
            else:
                resolved_params[key] = value if isinstance(value, list) else [value]

        # Erzeuge alle Kombinationen der Parameter
        keys, values = zip(*resolved_params.items())
        full_param_list.extend(dict(zip(keys, combination)) for combination in product(*values))

    return full_param_list


def run_variant(params):
    """Runs a single Variant"""
    cmd = ["python", "-m", params['module']]
    del params['module']

    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        if isinstance(value, list):
            cmd.append(f"--{key}")
            cmd.extend(map(str, value))
            continue
        cmd.extend([f"--{key}", str(value)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "params": params,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def batch_run(param_file, max_workers, num_tasks, task_id, verbose=False):
    """Führt nur den Teil der Varianten aus, der diesem Task zugeordnet ist."""

    params_list = load_params(param_file)
    total_variants = len(params_list)
    logger.info(f"Total {total_variants} Variants")

    # Teile die Parameterliste gleichmäßig auf
    chunk_size = (total_variants + num_tasks - 1) // num_tasks
    start_idx = task_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_variants)

    # Subset der Parameter für diesen Task
    params_subset = params_list[start_idx:end_idx]

    logger.debug(f"Task {task_id}: Running {len(params_subset)} Variants")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_variant, params_subset))

    for i, result in enumerate(results):
        logger.info(f"Variant {start_idx + i + 1} finished with return code {result['returncode']}")
        logger.debug(f"Standard Output:\n{result['stdout']}")
        logger.debug(f"Standard Error:\n{result['stderr']}")


if __name__ == "__main__":
    # Argumente für die Task-Nummer und Anzahl der Tasks
    parser = argparse.ArgumentParser(description="Parallel Batch Runner")
    parser.add_argument("--param_file", type=str, default="./config/params.json", help="Path to parameter file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum workers for subprocesses")
    parser.add_argument("--num_tasks", type=int, default=1, help="Total number of tasks")
    parser.add_argument("--task_id", type=int, default=None, help="Task ID (0-based index)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable detailed logging")
    args = parser.parse_args()

    # Überprüfen, ob task_id erforderlich ist
    if args.num_tasks != 1 and args.task_id is None:
        parser.error("--task_id is required when num_tasks != 1")

    if args.task_id is None:
        args.task_id = 0

    configure_logger(args.verbose)

    import multiprocessing
    multiprocessing.set_start_method('spawn')

    batch_run(
        args.param_file,
        args.max_workers,
        args.num_tasks,
        args.task_id,
        args.verbose
    )
