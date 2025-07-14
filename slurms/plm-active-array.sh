#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A naderilab
#SBATCH -p scavenger-gpu,gpu-common
#SBATCH -J slr_a_search
#SBATCH -o /hpc/group/naderilab/eleanor/Efficient_PLM/slurms/slr_a_search/%a.out
#SBATCH -e /hpc/group/naderilab/eleanor/Efficient_PLM/slurms/slr_a_search/%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=1
#SBATCH -t 2-00:00:00
#SBATCH --array=0-8
#

hostname
nvidia-smi
python /hpc/group/naderilab/eleanor/Efficient_PLM/train_prose_masked_tf.py \
  --clip 0.0001 \
  --batch-size 100 \
  --validate-every 10 \
  --plr 5e-5 \
  --cluster kmeans \
  --epsilon 2.6 \
  --nlayer 5 \
  --d-model 512 \
  --max-length 1024 \

