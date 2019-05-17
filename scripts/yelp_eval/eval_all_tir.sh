#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0
#SBATCH --array=0-3%3
##SBATCH --nodelist=compute-0-7

declare -a eval_list=("outputs_yelp_CVAE_newopt/yelp_wd0.0_wb0.0_ws0.0_an1_pool5_klw0.1_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_yelp_CVAE_newopt/yelp_wd0.0_wb0.0_ws0.0_an3_pool5_klw0.1_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_yelp_CVAE_newopt/yelp_wd0.1_wb0.2_ws3.0_an1_pool5_klw1.0_lr0.001_t0.01_bt/" \
    "outputs_yelp_CVAE_newopt/yelp_wd0.1_wb0.2_ws3.0_an2_pool5_klw1.0_lr0.001_t0.01_bt/")

arglen=${#eval_list[@]}

taskid=${SLURM_ARRAY_TASK_ID}

i=$(( taskid%arglen ))

./scripts/yelp_eval/eval_all.sh ${eval_list[$i]}
