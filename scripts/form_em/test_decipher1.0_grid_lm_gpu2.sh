#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

declare -a pool=("1" "5")
declare -a klw=("0.1" "0.3" "0.5")

for i in "${pool[@]}"
do
  for j in "${klw[@]}"
  do
    CUDA_VISIBLE_DEVICES=$1 python src/main.py \
      --dataset decipher1_0 \
      --clean_mem_every 5 \
      --reset_output_dir \
      --classifier_dir="pretrained_classifer/decipher" \
      --data_path data/test/ \
      --train_src_file data/yelp_decipher/yelp_decipher1.0/train.txt \
      --train_trg_file data/yelp_decipher/yelp_decipher1.0/train.attr \
      --dev_src_file data/yelp_decipher/yelp_decipher1.0/dev.txt \
      --dev_trg_file data/yelp_decipher/yelp_decipher1.0/dev.attr \
      --dev_trg_ref data/yelp_decipher/yelp_decipher1.0/dev_ref.txt \
      --src_vocab  data/yelp_decipher/yelp_decipher1.0/text.vocab \
      --trg_vocab  data/yelp_decipher/yelp_decipher1.0/attr.vocab \
      --d_word_vec=128 \
      --d_model=512 \
      --log_every=100 \
      --eval_every=3000 \
      --ppl_thresh=10000 \
      --eval_bleu \
      --batch_size 32 \
      --valid_batch_size 128 \
      --patience 5 \
      --lr_dec 0.8 \
      --lr 0.001 \
      --dropout 0.3 \
      --max_len 10000 \
      --seed 0 \
      --beam_size 1 \
      --word_blank 0. \
      --word_dropout 0. \
      --word_shuffle 0 \
      --cuda \
      --anneal_epoch 2 \
      --temperature 0.01 \
      --klw $j \
      --max_pool_k_size $i \
      --bt \
      --bt_stop_grad \
      --lm \
      # --avg_len \
      # --gs_soft \
  done
done
