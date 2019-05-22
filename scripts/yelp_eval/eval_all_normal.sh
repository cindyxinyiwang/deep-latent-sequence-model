#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0
#SBATCH --array=0-4%1
##SBATCH --nodelist=compute-0-7

declare -a eval_list=("outputs_yelp_CVAE_newopt/yelp_wd0.0_wb0.0_ws0.0_an1_pool5_klw0.01_lr0.001_t0.01_lm_bt_hard_avglen/")

export CUDA_VISIBLE_DEVICES=$1

for i in "${eval_list[@]}"
do
    ./scripts/yelp_eval/eval_all.sh $i > "${i}eval_out.log"
done
