#!/bin/sh
#SBATCH --job-name="v14live85"
#SBATCH --account="none
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=35000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=28:20:00
#SBATCH --output=./submit_logs/v14live85.text
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


echo "MyModel on LIVE 85 dist prediction "

for split_idx in {0..9};
do
    python src/trainer2.py \
        job=mymodel_v14_live \
        split_index="${split_idx}"
done >> submit_logs/mine_v14_live_fromKADIS_7_10k.log
#mine_v14_85per_live_onlyquali.log