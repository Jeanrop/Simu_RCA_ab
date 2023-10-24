#!/bin/bash
#SBATCH --time=02:59:59
#SBATCH --account=def-jeproa
#SBATCH --job-name=Conversion
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/jeanrop/scratch/out_job/%j.out

cd /home/jeanrop/Documents

module load python/3.9 
source env/env_convert/bin/activate

python Simu_RCA_ab/Python/Convert_full.py 