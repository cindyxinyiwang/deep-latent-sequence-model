#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
# export CUDA_VISIBLE_DEVICES="0"

python src/translate.py \
  --model_dir $1 \
  --test_src_file $2 \
  --test_trg_file $3 \
  --out_file $4 \
  --beam_size 1 \
  --merge_bpe \
  --cuda \
