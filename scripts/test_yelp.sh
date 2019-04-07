#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
# export CUDA_VISIBLE_DEVICES="0"

CUDA_VISIBLE_DEVICES=$1 python src/main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_yelp/yelp/" \
  --data_path data/test/ \
  --train_src_file data/yelp/train.txt \
  --train_trg_file data/yelp/train.attr \
  --dev_src_file data/yelp/dev.txt \
  --dev_trg_file data/yelp/dev.attr \
  --src_vocab  data/yelp/text.vocab \
  --trg_vocab  data/yelp/attr.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=100 \
  --eval_every=2500 \
  --ppl_thresh=15 \
  --batch_size 32 \
  --valid_batch_size=32 \
  --patience 5 \
  --lr_dec 0.8 \
  --dropout 0.3 \
  --max_len 10000 \
  --seed 0 \
  --word_blank 0.3 \
  --word_dropout 0.3 \
  --word_shuffle 1.3 \
  --cuda \
