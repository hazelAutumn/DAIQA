#!/bin/sh
#SBATCH --job-name="mv143livec"
#SBATCH --account="none
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=35000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:20:00
#SBATCH --output=./submit_logs/mv12ka.text
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


echo "MyModel on TID pretrain "

for split_idx in {0..3};
do
    python src/trainer2out.py \
        job=mymodel_v143_livec \
        split_index="${split_idx}"
done >> submit_logs/mine_v143_livec_2fromKADIS_7_10k.log
