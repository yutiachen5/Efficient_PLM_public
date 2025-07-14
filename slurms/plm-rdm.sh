#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A biostat
#SBATCH -p scavenger-gpu,gpu-common
#SBATCH -J PLM
#SBATCH -o /hpc/group/naderilab/eleanor/Efficient_PLM/slurms/rdm.out
#SBATCH -e /hpc/group/naderilab/eleanor/Efficient_PLM/slurms/rdm.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=1
#

export OPENBLAS_NUM_THREADS=1
python /hpc/group/naderilab/eleanor/Efficient_PLM/train_prose_masked_tf.py \
  --clip 0.0001 \
  --batch-size 100 \
  --validate-every 10 \
  --plr 5e-5 \
  --nlayer 5 \
  --d-model 512 \
  --name rdm \
  --max-length 1024 \
  --query-mode random \
  --epsilon 1000