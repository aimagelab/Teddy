#!/bin/bash
#SBATCH --gres=gpu:2080:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/Teddy/jobs/teddy_%j.out
#SBATCH --mem=32G
##SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10,vegeta,carabbaggio,germano,gervasoni,pippobaudo,rezzonico,ajeje,helmut,lurcanio
#SBATCH -J teddy

cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/Teddy || exit
# scontrol update JobID="$SLURM_JOB_ID" name="teddy_@{name}"
srun /homes/$(whoami)/.conda/envs/teddy/bin/python train.py --wandb --db_preload --weight_style @{style} --root_path /work/FoMo_AIISDH/datasets \
    --dis_patch_width @{pwidth} --dis_patch_count @{pcount}
