#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4096
#SBATCH --job-name=prob2lines
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/lanes/ERFNet-CULane/tools/prob2lines
#SBATCH --output=prob2lines.out
#SBATCH --error=prob2lines.err

module use /opt/insy/modulefiles
module load matlab/R2018b

srun matlab -nodisplay -r "main;exit" 
