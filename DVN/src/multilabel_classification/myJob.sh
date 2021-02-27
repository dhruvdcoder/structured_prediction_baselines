#!/bin/sh
#SBATCH --job-name=tn_high_reg
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

python3 dvn_multilabel.py