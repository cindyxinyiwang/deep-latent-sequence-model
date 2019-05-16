#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0
#SBATCH --array=0-4%1
##SBATCH --nodelist=compute-0-7

declare -a eval_list=("outputs_shakespeare_CVAE_newopt/shakespeare_wd0.1_wb0.2_ws3.0_an30_pool5_klw1.0_lr0.001_t0.01_bt/" \
    "outputs_shakespeare_CVAE_newopt/shakespeare_wd0.1_wb0.2_ws3.0_an50_pool5_klw1.0_lr0.001_t0.01_bt/" \
    "outputs_shakespeare_CVAE_newopt/shakespeare_wd0.1_wb0.2_ws3.0_an-1_pool5_klw1.0_lr0.001_t0.01_bt/")

export CUDA_VISIBLE_DEVICES=$1

for i in "${eval_list[@]}"
do
    ./scripts/shakespeare_eval/eval_all.sh $i > "${i}eval_out.log"
done
