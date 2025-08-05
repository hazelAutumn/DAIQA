#!/bin/sh
#SBATCH --job-name="mv1tidpre"
#SBATCH --account="none
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=35000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:20:00
#SBATCH --output=./submit_logs/mv1tidpre.text
#SBATCH --mail-user=email@
#SBATCH --mail-type=ALL

WORKDIR="code_directory"
cd ${WORKDIR}
echo " we are runing from this directory: $WORKDIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"



echo "MyModel on TID pretrain v2 with cross entropy loss "
# for split_idx in {0..1};
# do
#     python src/trainer.py \
#         job=mymodel2_kadid \
#         split_index="${split_idx}"
# done >> submit_logs/0my12real_kadid.log

# for split_idx in {0..4};
# do
#     python src/trainer2.py \
#         job=mymodel2_koniq \
#         split_index="${split_idx}"
# done >> submit_logs/0my_retrain_koniq.log

for split_idx in {0..2};
do
    python src/trainer2out.py \
        job=mymodel_tid2013 \
        split_index="${split_idx}"
done >> submit_logs/mine_v1_tid_pretrain_finetune_10.log