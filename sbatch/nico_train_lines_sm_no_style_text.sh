#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH -e /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.err
#SBATCH -o /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -J teddy
#SBATCH --account=fomo_aiisdh
#SBATCH --time=12:00:00
#SBATCH --constraint="gpu_A40_48G"

cd /work/FoMo_AIISDH/vpippi/Teddy || exit
# scontrol update JobID="$SLURM_JOB_ID" name="teddy"
srun /homes/$(whoami)/.conda/envs/teddy/bin/python train.py --batch_size 8 --root_path /work/FoMo_AIISDH/vpippi/Teddy/files/datasets/ --datasets iam_lines_sm --wandb --tag teddy_lines_sm_no_style --eval_dataset iam_eval --no_style_text  
