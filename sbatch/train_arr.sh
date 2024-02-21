#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.err
#SBATCH -o /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.out
#SBATCH --mem=32G
#SBATCH -J teddy_b4ce
#SBATCH --time=1-00:00:00
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --array=0-10%1

cd /work/FoMo_AIISDH/vpippi/Teddy || exit
# scontrol update JobID="$SLURM_JOB_ID" name="teddy"
/homes/$(whoami)/.conda/envs/teddy/bin/python -m nltk.downloader all
srun /homes/$(whoami)/.conda/envs/teddy/bin/python train.py --wandb --root_path /work/FoMo_AIISDH/vpippi/Teddy/files/datasets --batch_size 16 --tag def2 --resume --run_id b4ce
