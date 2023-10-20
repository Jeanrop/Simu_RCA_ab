#!/bin/bash
#SBATCH --time=05:59:59
#SBATCH --account=def-jeproa
#SBATCH --array=1-10
#SBATCH --job-name=Simu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/jeanrop/scratch/out_job/%j.out

cd /home/jeanrop/Documents/simu_ab

module load matlab/2023a

export SAVE_ID=$SLURM_ARRAY_TASK_ID
export Job_ID=$SLURM_JOB_ID

matlab -nodesktop -nodisplay -r "save_simus_sh"