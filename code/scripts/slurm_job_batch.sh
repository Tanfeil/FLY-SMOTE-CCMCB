#!/bin/bash
#SBATCH --job-name=fly-smote-parallel
#SBATCH --output=/bigwork/nhwast10/logs/job_output_%A/job_output_%A_%a.log   # Log-Dateien für jeden Task
#SBATCH --error=/bigwork/nhwast10/logs/job_error_%A/job_error_%A_%a.log     # Fehler-Log für jeden Task
#SBATCH --array=0-4                     # Array mit 10 Jobs (ID von 0 bis 9)
#SBATCH --cpus-per-task=32              # 4 CPUs pro Task
#SBATCH --mem-per-cpu=4G                # Speicher pro Task
#SBATCH --time=4:00:00                 # Maximale Laufzeit
#SBATCH --partition=enos
#SBATCH --mail-user=jonathan.feilmeier@stud.uni-hannover.de
#SBATCH --mail-type=ALL

module load Miniforge3
#conda create -n myenv python=3.11.5 numpy=1.26.4 pandas=2.2.3 requests scikit-learn tensorflow keras=3.5.0 tqdm wandb -y
conda activate myenv
cd $BIGWORK/FLY-SMOTE-CCMCB_adult_sensitivity

# Python-Skript für den aktuellen Task ausführen
python -m code.scripts.runner_parallel --param_file "./config/adult/params_sensitivity.json" --max_workers 3 --num_tasks 5 --task_id $SLURM_ARRAY_TASK_ID