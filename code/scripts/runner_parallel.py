"""
This module provides functionality for running parameter variants in parallel across multiple tasks. It loads parameter combinations from a JSON file, executes the variants in separate processes, and logs the results.

Functions:
- load_parameters: Load and process parameter combinations from a JSON file.
- execute_variant: Execute a single parameter variant using subprocesses.
- run_in_batches: Distribute the work of running parameter variants across multiple tasks in parallel.
"""

import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product

import numpy as np

from code.shared.logger_config import setup_logger

# Set up logging configuration
logger = logging.getLogger()


def load_parameters(param_file):
    """
    Load parameters from a JSON file and generate all combinations.

    Args:
        param_file (str): Path to the JSON file containing parameter definitions.

    Returns:
        list: A list of dictionaries representing all possible combinations of parameters.

    Example:
        load_parameters("./config/params.json")
    """
    with open(param_file, "r") as file:
        raw_parameters = json.load(file)

    all_parameter_combinations = []

    for param_set in raw_parameters:
        resolved_parameters = {}

        # Resolve ranges and permutations in parameter sets
        for param_name, param_value in param_set.items():
            if isinstance(param_value, dict):
                if "range" in param_value:
                    start, stop, step = param_value["range"]
                    resolved_parameters[param_name] = np.arange(start, stop + step, step).tolist()
                elif "perm" in param_value:
                    numbers = param_value["perm"]
                    param_combinations = [
                        sorted(comb, reverse=param_value["reverse"]) for r in range(1, len(numbers) + 1)
                        for comb in combinations(numbers, r)
                    ]
                    resolved_parameters[param_name] = param_combinations
            else:
                resolved_parameters[param_name] = param_value if isinstance(param_value, list) else [param_value]

        # Generate all combinations of the resolved parameters
        keys, values = zip(*resolved_parameters.items())
        all_parameter_combinations.extend(dict(zip(keys, combination)) for combination in product(*values))

    return all_parameter_combinations


def execute_variant(parameters):
    """
    Run a single variant using the provided parameters.

    Args:
        parameters (dict): A dictionary containing parameter names as keys and their corresponding values.

    Returns:
        dict: A dictionary with the following keys:
            - "params": The input parameters.
            - "returncode": The return code of the process.
            - "stdout": The standard output from the process.
            - "stderr": The standard error from the process.

    Example:
        execute_variant({'module': 'my_module', 'param1': 10, 'param2': [1, 2]})
    """
    command = ["python", "-m", parameters['module']]
    del parameters['module']

    for param_name, param_value in parameters.items():
        if isinstance(param_value, bool):
            if param_value:
                command.append(f"--{param_name}")
        elif isinstance(param_value, list):
            command.append(f"--{param_name}")
            command.extend(map(str, param_value))
        else:
            command.extend([f"--{param_name}", str(param_value)])

    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "params": parameters,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_in_batches(param_file, max_workers, total_tasks, task_id, verbose=False):
    """
    Execute the parameter variants assigned to this specific task in parallel batches.

    Args:
        param_file (str): Path to the parameter file containing the parameter combinations.
        max_workers (int): The maximum number of workers for parallel processing.
        total_tasks (int): The total number of tasks to split the work across.
        task_id (int): The task ID (0-based index) to identify the specific task's assigned parameters.
        verbose (bool, optional): Flag to enable detailed logging. Defaults to False.

    Returns:
        None: This function only logs the progress and results of the executed variants.

    Example:
        run_in_batches("./config/params.json", 4, 10, 0, verbose=True)
    """
    parameter_combinations = load_parameters(param_file)
    total_variants = len(parameter_combinations)
    logger.info(f"Total {total_variants} Variants to process")

    # Split the parameters into roughly equal chunks across tasks
    chunk_size = (total_variants + total_tasks - 1) // total_tasks
    start_idx = task_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_variants)

    # Get the subset of parameters assigned to this task
    parameters_for_task = parameter_combinations[start_idx:end_idx]

    logger.debug(f"Task {task_id}: Running {len(parameters_for_task)} Variants")

    with ProcessPoolExecutor(max_workers=max_workers, initializer=setup_logger,
                             initargs=(verbose,)) as executor:
        results = list(executor.map(execute_variant, parameters_for_task))

    # Log the results for each variant processed
    for i, result in enumerate(results):
        logger.info(f"Variant {start_idx + i + 1} finished with return code {result['returncode']}")
        logger.debug(f"Standard Output:\n{result['stdout']}")
        logger.debug(f"Standard Error:\n{result['stderr']}")


if __name__ == "__main__":
    # Argument parser for the task execution
    parser = argparse.ArgumentParser(description="Parallel Batch Runner")
    parser.add_argument("--param_file", type=str, default="./config/params.json", help="Path to parameter file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of workers for subprocesses")
    parser.add_argument("--total_tasks", type=int, default=1, help="Total number of tasks to split the work")
    parser.add_argument("--task_id", type=int, default=None, help="Task ID (0-based index)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable detailed logging")
    args = parser.parse_args()

    # Ensure task_id is provided when num_tasks > 1
    if args.total_tasks != 1 and args.task_id is None:
        parser.error("--task_id is required when total_tasks != 1")

    if args.task_id is None:
        args.task_id = 0

    # Set up logging configuration
    setup_logger(args.verbose)

    import multiprocessing

    multiprocessing.set_start_method('spawn')

    # Run the batch process with the provided arguments
    run_in_batches(
        args.param_file,
        args.max_workers,
        args.total_tasks,
        args.task_id,
        args.verbose
    )
