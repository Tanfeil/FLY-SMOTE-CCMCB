# FLY-SMOTE-CCMCB

This Repository implements the 

## The data used in this project:

* [Hotels](https://github.com/upcbdipt/CDW_FedAvg/tree/main)
* [Adult](https://archive.ics.uci.edu/ml/datasets/adult)
* [Compass](https://www.kaggle.com/datasets/danofer/compass)
* [Bank](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

## Configurations

Configurations for different Setups, can be found as JSON Files under config

## Code

The code is divided as follows:

* The code.FLY-SMOTE-CCMCB.main python file contains the necessary code to run an experiement.
* The code.shared.FlySmote contains the necessary functions to apply fly-smote re-balancing method.
* the code.shared.NNModel contains the neural network model.
* The code.shared.GAN contains the CGAN
* The code.shared.DatasetLoader file contains the necessary functions to read the datasets.
* The code.shared.train_client holds functions to train one client
* All other modules hold helper functions or configurations functions
* In Scripts can be found different scripts, to create plots, run FLY-SMOTE-CCMCB by JSON configs or start it as a SLURM Job on Clusters.
* dataset/hotels contains raw and processed data of the dataset from (https://github.com/upcbdipt/CDW_FedAvg/tree/main)

A documentation can be found under: https://tanfeil.github.io/FLY-SMOTE-CCMCB/

## Environment Setup
To use this code are several steps needed

### Conda setup
```bash
conda create -n myenv python=3.11.5
conda activate myenv

pip install -r requirements.txt
```

### Load Datasets into dataset
```bash
python -m code.scripts.download_datasets
```

## Run FLY-SMOTE-CCMCB
There are different options to run the method:

### From CLI
```bash
python -m code.FLY-SMOTE-CCMCB.main -d <dataset> -f <datasetpath>
```
All other parameters have settings by default. To see the parameters just call 
```bash
python -m code.FLY-SMOTE-CCMCB.main -h
```
Examples:
- FLY-SMOTE-CCMCB
```bash
python -m code.FLY-SMOTE-CCMCB.main -d <dataset> -f <datasetpath> --ccmcb
```
- FLY-SMOTE
```bash
python -m code.FLY-SMOTE-CCMCB.main -d <dataset> -f <datasetpath> 
```
-FdgAvg
```bash
python -m code.FLY-SMOTE-CCMCB.main -d <dataset> -f <datasetpath> --threshold 0
```
Change the other params like k, r, g if you need to

### With a JSON Config
If you want to use some predefined configurations you can run
```bash
python -m code.scripts.runner_parallel --param_file <path to config> --max_workers 3 --num_tasks 5
```
To see all parameters call
```bash
python -m code.scripts.runner_parallel -h
```
### Run a SLURM job
To run configuration as a SLURM job on a cluster the call is
```bash
sbatch code/scripts/slurm_job_batch.sh
```

Note that you have to adjust the SLURM settings to your cluster and directory configurations.
### Plots and wandb project names
For compatibility with scripts for generating plots and other things, please name wandb projects as follows: <dataset>_<some run specification of yours>

### Generate Documentation
pydoctor --make-html --html-output=docs --theme=readthedocs --project-name="FLY-SMOTE-CCMCB" --intersphinx=https://docs.python.org/3/objects.inv ./code

## Prerequisites

The python packages needed, as stated in requirements.txt are:

* keras==3.5.0
* matplotlib==3.10.1
* numpy==1.26.4
* pandas==2.2.3
* requests==2.32.3
* scikit-learn==1.6.1
* seaborn==0.13.2
* tensorflow==2.17.0
* tqdm==4.67.1
* wandb==0.19.7

## Reference

If you re-use this work, please cite:

