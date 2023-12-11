#!/bin/bash
#SBATCH -n 32                  # Number of cores requested
#SBATCH -t 0-12:00:00          # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH -p seas_gpu            # Partition to submit to
#SBATCH --mem=64000            # Memory per core in MB
#SBATCH --gres=gpu:1           # Number of GPUs to use
#SBATCH -o ./%j.log  # File to which output and errors will be written, %j inserts jobid

source venv/bin/activate
python model_training.py
