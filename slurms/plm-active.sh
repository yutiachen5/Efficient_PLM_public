#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A h200ea
#SBATCH -p h200ea
#SBATCH -J PLM
#SBATCH -o /hpc/group/naderilab/eleanor/Efficient_PLM/slurms/esm_nhead_4_plr_5e-5.out
#SBATCH -e /hpc/group/naderilab/eleanor/Efficient_PLM/slurms/esm_nhead_4_plr_5e-5.err
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
  --plr 5e-5 \
  -n 2000 \
  --cluster kmeans \
  --epsilon 2.6 \
  --nlayer 6 \
  --d-model 320 \
  --weight-decay 0.01 \
  --nhead 4 \
  --name esm_nhead_6_plr_5e-5 \
  --max-length 1024 \
  --alpha-slack 0.1 \
  --lr-slack 0.05 \
  --encoding RoPE
