#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.err
#SBATCH -o /work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.out
#SBATCH --mem=32G
#SBATCH -J teddy
#SBATCH --time=1-00:00:00
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G"

cd /work/FoMo_AIISDH/vpippi/Teddy || exit
# scontrol update JobID="$SLURM_JOB_ID" name="teddy"
srun /homes/$(whoami)/.conda/envs/teddy/bin/python train.py --batch_size 16 --style_patch_width 64 --root_path /home/vpippi/Teddy/files/datasets/ --datasets iam_words --dryrun --tag teddy_words --gen_emb_module OnehotModule --eval_dataset iam_eval_words  
