#!/bin/bash
#SBATCH --time=23:50:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --job-name=lbd_null_pos
#SBATCH --output=out-%x-%j.out

# Display date
echo "DATE: $(date)"

# Display sbatch script
echo "SBATCH SCRIPT:"
echo "$(cat job_realigned_multi.sh)"

# Display output
echo "SCRIPT OUTPUT:"

# Load modules
module load StdEnv/2020  gcc/9.3.0  cuda/11.0  openmpi/4.0.3
module load arrayfire/3.7.3
module load python/3.7
#source $SLURM_TMPDIR/env/bin/activate
source /home/$USER/environments/env/bin/activate

#pip install numpy matplotlib scipy h5py pdbpp
#pip install --no-index --upgrade pip
#pip install --no-index arrayfire



# Execute script

#exemple of script used on compute canada

#for lbd/2
python ./generate_data_high_speckle_randn.py  250 0.5 0.0 data_realigned_high_pos/train_set/realigned_lambda_2_amp_00_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.5 0.1 data_realigned_high_pos/train_set/realigned_lambda_2_amp_10_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.5 0.2 data_realigned_high_pos/train_set/realigned_lambda_2_amp_20_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.5 0.3 data_realigned_high_pos/train_set/realigned_lambda_2_amp_30_pct_pos

#for lbd/4
python ./generate_data_high_speckle_randn.py  250 0.25 0.0 data_realigned_high_pos/train_set/realigned_lambda_4_amp_00_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.25 0.1 data_realigned_high_pos/train_set/realigned_lambda_4_amp_10_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.25 0.2 data_realigned_high_pos/train_set/realigned_lambda_4_amp_20_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.25 0.3 data_realigned_high_pos/train_set/realigned_lambda_4_amp_30_pct_pos

#for lbd/8
python ./generate_data_high_speckle_randn.py  250 0.125 0.0 data_realigned_high_pos/train_set/realigned_lambda_8_amp_00_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.125 0.1 data_realigned_high_pos/train_set/realigned_lambda_8_amp_10_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.125 0.2 data_realigned_high_pos/train_set/realigned_lambda_8_amp_20_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.125 0.3 data_realigned_high_pos/train_set/realigned_lambda_8_amp_30_pct_pos

#for 0 lbd
python ./generate_data_high_speckle_randn.py  250 0.0 0.0 data_realigned_high_pos/train_set/realigned_lambda_null_amp_00_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.0 0.1 data_realigned_high_pos/train_set/realigned_lambda_null_amp_10_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.0 0.2 data_realigned_high_pos/train_set/realigned_lambda_null_amp_20_pct_pos
python ./generate_data_high_speckle_randn.py  250 0.0 0.3 data_realigned_high_pos/train_set/realigned_lambda_null_amp_30_pct_pos
