#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --mail-user=email
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --time=04-23:00:00
#SBATCH --account=account
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=48000m

#SBATCH --output=./logs/PASCAL_human_parts_step2.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /user/Codes/Pascal_Train_VIT/human_parts/Step2.py --gpu 0,1,2,3 --dist_on_itp False --WORLD_SIZE 4 --LOCAL_RANK 0 --RANK 0
