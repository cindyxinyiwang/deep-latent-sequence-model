#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
# export CUDA_VISIBLE_DEVICES="0"

CUDA_VISIBLE_DEVICES=$1 python src/translate.py \
  --model_dir outputs_yelp/yelp_wd0.2_wb0.2_ws1.2_an1_gs1.0/ \
  --test_src_file data/yelp/test_sub.txt \
  --test_trg_file data/yelp/test_sub.attr \
  --beam_size 1 \
  --cuda \
