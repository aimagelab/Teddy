#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.err
#SBATCH -o /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.out
#SBATCH --mem=32G
#SBATCH -J ocr0
#SBATCH --account=fomo_aiisdh
#SBATCH --time=1-00:00:00
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --array=0-10%1

cd /work/FoMo_AIISDH/vpippi/Teddy || exit
# scontrol update JobID="$SLURM_JOB_ID" name="teddy"
srun /homes/$(whoami)/.conda/envs/teddy/bin/python train.py --root_path /work/FoMo_AIISDH/vpippi/Teddy/files/datasets/ --wandb --tag eccv_train_ocr --resume --run_id ocr0 --ocr_scheduler train