#!/bin/bash
#SBATCH -A zhangmlgroup
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:6
#SBATCH -N 2
#SBATCH --cpus-per-task=20
#SBATCH --time=30:00:00

# Load any modules and activate your conda environment here
module load singularity sgp
#WORKDIR=/opt/stochastic_gradient_push

#/scratch/$USER/sgp-pytorch-1.0.sif 
srun singularity run --nv $CONTAINERDIR/sgp-pytorch-1.0.sif  \
    -u gossip_sgd1.0.py \
    --checkpoint_dir=log \
    --dataset_dir=tiny-imagenet-200 \
    --batch_size 256 --lr 0.1 --num_dataloader_workers $((SLURM_CPUS_PER_TASK/2)) \
    --num_epochs 90 --nesterov True --warmup True --push_sum False \
    --schedule 30 0.1 60 0.1 80 0.1 \
    --train_fast False --master_port 40100 \
    --tag 'AR-SGD-IB' --print_freq 100 --verbose False \
    --graph_type -1 --all_reduce True --seed 1 \
    --network_interface_type 'infiniband'
