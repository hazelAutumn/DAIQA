#!/bin/sh
#SBATCH --job-name="infere"
#SBATCH --account="none
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:20:00
#SBATCH --output=./submit_logs/0inferer.text
#SBATCH --mail-user=email@
#SBATCH --mail-type=ALL

WORKDIR="code_directory"
cd ${WORKDIR}
echo " we are runing from this directory: $WORKDIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.1.1


echo "Inference single image"


for split_idx in {0..1..2};
do
    python src/inference.py \
        job=inference_single
done >> submit_logs/000inference_81.log #2>&1


