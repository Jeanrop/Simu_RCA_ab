#!/bin/bash
#SBATCH --time=7-00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --job-name=crelu_high_speckle_randn
#SBATCH --output=out-%x-%j.out

# Display date
echo "DATE: $(date)"

# Display sbatch script
echo "SBATCH SCRIPT:"
echo "$(cat job_IQ.sh)"

# Display output
echo "SCRIPT OUTPUT:"

# Load modules
module load StdEnv/2020  gcc/9.3.0  cuda/11.0  openmpi/4.0.3
module load arrayfire/3.7.3
module load python/3.7
# Load environment
source /home/$USER/environments/env_af/bin/activate



# Execute script
python IQtrain.py --config=config/config.json 
