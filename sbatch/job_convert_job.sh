#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --account=def-jeproa
#SBATCH --array=1-10
#SBATCH --job-name=Conversion_job
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/jeanrop/scratch/out_job/%j.out

cd /home/jeanrop/Documents

module load python/3.9 
source env/env_convert/bin/activate

python Simu_RCA_ab/Python/Convert_job.py $SLURM_ARRAY_TASK_ID