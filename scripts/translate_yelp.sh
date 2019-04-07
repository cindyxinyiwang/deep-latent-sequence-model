#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
# export CUDA_VISIBLE_DEVICES="0"

CUDA_VISIBLE_DEVICES=$1 python src/translate.py \
  --model_dir outputs_yelp/yelp/ \
  --test_src_file data/yelp/test_sub.txt \
  --test_trg_file data/yelp/test_sub.attr \
  --beam_size 5 \
  --cuda \
