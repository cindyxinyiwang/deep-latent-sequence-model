#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0
#SBATCH --array=0-4%1
##SBATCH --nodelist=compute-0-7

declare -a eval_list=("outputs_form_em_CVAE_newopt_dual/form_em_wd0.1_wb0.2_ws3.0_an3_pool5_klw0.03_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_form_em_CVAE_newopt_dual/form_em_wd0.1_wb0.2_ws3.0_an-1_pool5_klw0.1_lr0.001_t0.01_lm_bt_hard_avglen/" \
    "outputs_form_em_CVAE_newopt_dual/form_em_wd0.1_wb0.2_ws3.0_an-1_pool5_klw0.03_lr0.001_t0.01_lm_bt_hard_avglen/")

export CUDA_VISIBLE_DEVICES=$1

for i in "${eval_list[@]}"
do
    ./scripts/form_em_eval/eval_all.sh $i > "${i}eval_out.log"
done
