#!/bin/sh
#SBATCH --job-name="dist"
#SBATCH --account="none
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=35000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:40:00
#SBATCH --output=text
#SBATCH --mail-user=email@
#SBATCH --mail-type=ALL



module purge
module load Anaconda3/2023.09-0
module load CUDA/12.1.1




for split_idx in {7..9};
do
    python src/trainer.py \
        job=train_dist_tid2013 \
        split_index="${split_idx}"
done >> logs_new/dist_tid2013.log #2>&1