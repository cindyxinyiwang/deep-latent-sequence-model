#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"

python src/lm_lstm.py \
    --dataset $1 \
    --eval_from $2 \
    --test_src_file $3 \
    --test_trg_file $4 \
    --style 0 \