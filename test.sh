#!/bin/sh
#SBATCH --partition=visionlab --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=16384
#SBATCH --job-name=lane
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/lanes/ERFNet-CULane
#SBATCH --output=test.out
#SBATCH --error=test.err

module use /opt/insy/modulefiles
module load cuda/10.1  cudnn/10.1-7.6.0.64
srun nvidia-smi

./test_erfnet.sh
