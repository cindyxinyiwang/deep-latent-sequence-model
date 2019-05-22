#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export CUDA_VISIBLE_DEVICES=0

data_dir="data/yelp"
dataset="yelp"
model0_dir=$(printf "pretrained_lm/%s_style0/model.pt" $dataset)
model1_dir=$(printf "pretrained_lm/%s_style1/model.pt" $dataset)

test_src_file0=$data_dir/test_0.txt
test_src_file1=$data_dir/test_1.txt
test_trg_file0=$data_dir/test_0.attr
test_trg_file1=$data_dir/test_1.attr

printf "\neval text style0 under model style 0\n"
./scripts/sanity_check_lm.sh $dataset $model0_dir $test_src_file0 $test_trg_file0

printf "\neval text style1 under model style 0\n"
./scripts/sanity_check_lm.sh $dataset $model0_dir $test_src_file1 $test_trg_file1

printf "\neval text style0 under model style 1\n"
./scripts/sanity_check_lm.sh $dataset $model1_dir $test_src_file0 $test_trg_file0

printf "\neval text style1 under model style 1\n"
./scripts/sanity_check_lm.sh $dataset $model1_dir $test_src_file1 $test_trg_file1
