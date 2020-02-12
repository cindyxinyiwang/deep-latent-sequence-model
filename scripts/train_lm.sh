#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES=$1 python src/lm_lstm.py \
    --dataset yelp \
    --style 0