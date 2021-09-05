#!/bin/bash
#SBATCH -A zhangmlgroup
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:4
#SBATCH -N 2
#SBATCH --cpus-per-task=20
#SBATCH --time=30:00:00
#SBATCH --output=batch_size_%a.out
#SBATCH -a 1-6
# Load any modules and activate your conda environment here
module load singularity sgp

BS=(32 64 128 256 512 1024)
OUTDIR=bs_result_$SLURM_ARRAY_TASK_ID

#WORKDIR=/opt/stochastic_gradient_push
#/scratch/$USER/sgp-pytorch-1.0.sif 
srun singularity run --nv $CONTAINERDIR/sgp-pytorch-1.0.sif  \
    -u gossip_sgd1.0.py \
    --checkpoint_dir $OUTDIR \
    --dataset_dir=tiny-imagenet-200 \
    --batch_size ${BS[$((SLURM_ARRAY_TASK_ID-1))]} --lr 0.1 --num_dataloader_workers $((SLURM_CPUS_PER_TASK/2)) \
    --num_epochs 90 --nesterov True --warmup True --push_sum False \
    --schedule 30 0.1 60 0.1 80 0.1 \
    --train_fast False --master_port $((40100+SLURM_ARRAY_TASK_ID)) \
    --tag 'AR-SGD-IB' --print_freq 100 --verbose False \
    --graph_type -1 --all_reduce True --seed 1 \
    --network_interface_type 'infiniband'
