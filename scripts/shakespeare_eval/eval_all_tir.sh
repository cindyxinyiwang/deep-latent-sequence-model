#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0
#SBATCH --array=0-5%4
#SBATCH --exclude=compute-0-15,compute-0-17
##SBATCH --nodelist=compute-0-7

declare -a eval_list=("outputs_shakespeare_CVAE_newopt/shakespeare_wd0.0_wb0.0_ws0.0_an30_pool5_klw0.05_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_shakespeare_CVAE_newopt/shakespeare_wd0.0_wb0.0_ws0.0_an50_pool5_klw0.1_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_shakespeare_CVAE_newopt/shakespeare_wd0.0_wb0.0_ws0.0_an-1_pool5_klw0.1_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_shakespeare_CVAE_newopt_dual/shakespeare_wd0.0_wb0.0_ws0.0_an5_pool5_klw0.01_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_shakespeare_CVAE_newopt_dual/shakespeare_wd0.0_wb0.0_ws0.0_an30_pool5_klw0.05_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_shakespeare_CVAE_newopt_dual/shakespeare_wd0.0_wb0.0_ws0.0_an-1_pool5_klw0.05_lr0.001_t0.01_lm_bt_hard_avglen/")

arglen=${#eval_list[@]}

taskid=${SLURM_ARRAY_TASK_ID}

i=$(( taskid%arglen ))

./scripts/shakespeare_eval/eval_all.sh ${eval_list[$i]} > "${eval_list[i]}eval_out.log"
