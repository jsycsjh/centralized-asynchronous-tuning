#!/bin/bash
#SBATCH -A zhangmlgroup
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:4
#SBATCH -N 2
#SBATCH --cpus-per-task=20
#SBATCH --time=30:00:00
#SBATCH --output=learning_rate_%a.out
#SBATCH -a 1-14
# Load any modules and activate your conda environment here
module load singularity sgp

LR=(0.04 0.045 0.05 0.06 0.07 0.08 0.09)
OUTDIR=large_lr_result_$SLURM_ARRAY_TASK_ID

#WORKDIR=/opt/stochastic_gradient_push
#/scratch/$USER/sgp-pytorch-1.0.sif 
srun singularity run --nv $CONTAINERDIR/sgp-pytorch-1.0.sif  \
    -u gossip_sgd1.0.py \
    --checkpoint_dir $OUTDIR \
    --dataset_dir=tiny-imagenet-200 \
    --batch_size 256 --lr ${LR[$((SLURM_ARRAY_TASK_ID-1))]} --num_dataloader_workers $((SLURM_CPUS_PER_TASK/2)) \
    --num_epochs 90 --nesterov True --warmup True --push_sum False \
    --schedule 30 0.1 60 0.1 80 0.1 \
    --train_fast False --master_port $((40100+SLURM_ARRAY_TASK_ID)) \
    --tag 'AR-SGD-IB' --print_freq 100 --verbose False \
    --graph_type -1 --all_reduce True --seed 1 \
    --network_interface_type 'infiniband'
