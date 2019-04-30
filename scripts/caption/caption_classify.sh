#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

CUDA_VISIBLE_DEVICES=$1 python src/cnn_classify.py \
  --dataset caption \
  --classifer lstm \
  --output_dir "outputs_caption_classify" \
  --clean_mem_every 5 \
  --reset_output_dir \
  --data_path data/test/ \
  --train_src_file data/caption/train.txt \
  --train_trg_file data/caption/train.attr \
  --dev_src_file data/caption/dev.txt \
  --dev_trg_file data/caption/dev.attr \
  --dev_trg_ref data/caption/dev.txt \
  --src_vocab  data/caption/text.vocab \
  --trg_vocab  data/caption/attr.vocab \
  --d_word_vec=64 \
  --d_model=256 \
  --log_every=100 \
  --eval_every=1500 \
  --ppl_thresh=10000 \
  --eval_bleu \
  --out_c_list="1,2,3,4" \
  --k_list="3,3,3,3" \
  --batch_size 32 \
  --valid_batch_size=32 \
  --patience 5 \
  --lr_dec 0.8 \
  --dropout 0.5 \
  --max_len 10000 \
  --seed 0 \
  --beam_size 1 \
  --word_blank 0.1 \
  --word_dropout 0.1 \
  --word_shuffle 0 \
  --cuda \
  --temperature 0.01 \
  # --gumbel_softmax \
