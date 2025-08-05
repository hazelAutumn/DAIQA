#!/bin/sh
#SBATCH --job-name="process"
#SBATCH --account="share-none
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:40:00
#SBATCH --output=./submit_logs/0process.text
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


echo "Process kadis"


# dataset_names=("live" "tid2013" "kadid10k" "livechallenge" "koniq10k" "spaq" "flive")
# for dn in "${dataset_names[@]}"
# do
#     python scripts/process_"$dn".py
# done


python scripts/process_kadid10k.py
