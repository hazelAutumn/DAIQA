#!/bin/sh
#SBATCH --job-name="mv14ka80"
#SBATCH --account="none"
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=35000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=28:20:00
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


echo "MyModel on Kadid v14 from KADIS 5000 "


for split_idx in {0..9};
do
    python src/trainer2.py \
        job=mymodel_v14_kadid \
        split_index="${split_idx}"
done >> submit_logs/mine_v14_kadid_fr_KADIS7_10k.log
#submit_logs/mine_v14_kadid_fr_KADIS25_10000.log

########### For the accuracy of distortion classification
# for split_idx in {0..6};
# do
#     python src/eval.py \
#         job=mymodel_v14_kadid \
#         split_index="${split_idx}"
# done >> submit_logs/mine_v14_85per_kadid_eachdis.log