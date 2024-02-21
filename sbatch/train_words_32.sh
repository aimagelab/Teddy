#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.err
#SBATCH -o /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.out
#SBATCH --mem=32G
#SBATCH -J teddy
#SBATCH --account=fomo_aiisdh
#SBATCH --time=1-00:00:00
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G"

cd /work/FoMo_AIISDH/vpippi/Teddy || exit
# scontrol update JobID="$SLURM_JOB_ID" name="teddy"
/homes/$(whoami)/.conda/envs/teddy/bin/python -m nltk.downloader all
srun /homes/$(whoami)/.conda/envs/teddy/bin/python train.py --batch_size 8 --style_patch_width 32 --root_path /work/FoMo_AIISDH/vpippi/Teddy/files/datasets/ --datasets iam_words --wandb --tag teddy_words_32 --eval_dataset iam_eval_words  
