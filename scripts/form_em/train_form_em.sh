#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0



python src/main.py \
        --dataset form_em \
        --clean_mem_every 5 \
        --reset_output_dir \
        --classifier_dir="pretrained_classifer/form_em" \
        --train_src_file data/form_em/train.txt \
        --train_trg_file data/form_em/train.attr \
        --dev_src_file data/form_em/dev.txt \
        --dev_trg_file data/form_em/dev.attr \
        --dev_trg_ref data/form_em/dev.txt \
        --src_vocab  data/form_em/text.vocab \
        --trg_vocab  data/form_em/attr.vocab \
        --d_word_vec=128 \
        --d_model=512 \
        --log_every=100 \
        --eval_every=1500 \
        --ppl_thresh=10000 \
        --eval_bleu \
        --batch_size 16 \
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
