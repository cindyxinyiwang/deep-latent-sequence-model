#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0
#SBATCH --array=0-0%1
##SBATCH --nodelist=compute-0-7

python src/main.py \
  --dataset yelp \
  --clean_mem_every 5 \
  --reset_output_dir \
  --classifier_dir="pretrained_classifer/yelp" \
  --train_src_file data/yelp/train.txt \
  --train_trg_file data/yelp/train.attr \
  --dev_src_file data/yelp/dev_li.txt \
  --dev_trg_file data/yelp/dev_li.attr \
  --dev_trg_ref data/yelp/dev_li.txt \
  --src_vocab  data/yelp/text.vocab \
  --trg_vocab  data/yelp/attr.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=100 \
  --eval_every=1500 \
  --ppl_thresh=10000 \
  --eval_bleu \
  --batch_size 32 \
  --valid_batch_size 128 \
  --patience 5 \
  --lr_dec 0.5 \
  --lr 0.001 \
  --dropout 0.3 \
  --max_len 10000 \
  --seed 0 \
  --beam_size 1 \
  --word_blank 0. \
  --word_dropout 0. \
  --word_shuffle 0. \
  --cuda \
  --anneal_epoch 3 \
  --temperature 0.01 \
  --max_pool_k_size 5 \
  --bt \
  --bt_stop_grad \
  --klw 0.1 \
  --lm \
  --avg_len \
