#!/bin/bash

# SPDX-FileCopyrightText: 2025 Jonathan Feilmeier
#
# SPDX-License-Identifier: GPL-3.0-or-later

#SBATCH --job-name=fly-smote-parallel
#SBATCH --output=/bigwork/nhwast10/logs/job_output_%A/job_output_%A_%a.log
#SBATCH --error=/bigwork/nhwast10/logs/job_error_%A/job_error_%A_%a.log
#SBATCH --array=0-4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --mail-user=jonathan.feilmeier@stud.uni-hannover.de
#SBATCH --mail-type=ALL

module load Miniforge3
#conda create -n myenv python=3.11.5 numpy=1.26.4 pandas=2.2.3 requests scikit-learn tensorflow keras=3.5.0 tqdm wandb -y
conda activate myenv
cd $BIGWORK/FLY-SMOTE-CCMCB_compass_multiple

# Python-Skript für den aktuellen Task ausführen
python -m code.scripts.runner_parallel --param_file "./config/compass/params_multiple.json" --max_workers 3 --total_tasks 5 --task_id $SLURM_ARRAY_TASK_ID