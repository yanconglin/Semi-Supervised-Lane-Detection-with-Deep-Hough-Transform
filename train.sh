#!/bin/sh
#SBATCH --partition=visionlab --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=4096
#SBATCH --job-name=semi1

module use /opt/insy/modulefiles
module load cuda/10.1  cudnn/10.1-7.6.0.64
module load matlab/R2018b
srun nvidia-smi
pwd
./train_erfnet.sh
./test_erfnet2.sh

# cd /tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/lanes/tools/prob2lines
# pred_folder='/tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/lanes/ERFNet-CULane-HTIHT-semi/predicts'
# out_folder=${pred_folder}/output/
# pred_folder=${pred_folder}/
# matlab -nodisplay -r "prob2lines ${pred_folder} ${out_folder};quit"

