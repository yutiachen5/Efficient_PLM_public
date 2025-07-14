#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A h200ea
#SBATCH -p h200ea
#SBATCH -J PLM
#SBATCH -o /hpc/group/naderilab/eleanor/Efficient_PLM/slurms/esm_architecture.out
#SBATCH -e /hpc/group/naderilab/eleanor/Efficient_PLM/slurms/esm_architecture.err
#SBATCH --exclusive
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=1
#SBATCH -t 7-00:00:00
#

hostname
nvidia-smi
python /hpc/group/naderilab/eleanor/Efficient_PLM/train_prose_masked_tf.py \
  --clip 0.0001 \
  --batch-size 100 \
  --validate-every 10 \
  --plr 4e-4 \
  --nsteps 10000 \
  --cluster kmeans \
  --epsilon 2.6 \
  --nlayer 6 \
  --d-model 320 \
  --weight-decay 0.01 \
  --nhead 20 \
  --name esm_architecture \
  --max-length 1024 \
  --alpha-slack 0.1 \
  --lr-slack 0.05 \
