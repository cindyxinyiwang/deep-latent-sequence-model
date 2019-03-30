#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3.6 src/main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_test/test/" \
  --data_path data/test/ \
  --train_src_file data/test/train.txt \
  --train_trg_file data/test/train.attr \
  --dev_src_file data/test/train.txt \
  --dev_trg_file data/test/train.attr \
  --src_vocab  data/test/train.txt.vocab \
  --trg_vocab  data/test/train.attr.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=15 \
  --cuda \
  --batcher='word' \
  --batch_size 1500 \
  --valid_batch_size=7 \
  --patience 5 \
  --lr_dec 0.8 \
  --dropout 0.3 \
  --max_len 10000 \
  --seed 0
