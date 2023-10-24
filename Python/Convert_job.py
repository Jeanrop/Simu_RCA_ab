#%% Initialization
from scipy.io import loadmat 
import h5py
import numpy as np
import os
import time


Path_save = r'/home/jeanrop/Documents/data/h5_job'
Path_load = r'/home/jeanrop/scratch/10_job_100_simu_2'
Path_ini  = os.path.join(Path_load, str(1), str(1))

var_names = [file[:-4] for file in os.listdir(Path_ini)]

# Help to initialize the size of the h5 file
job_folders = [folder for folder in os.listdir(Path_load)]
# Help to initialize the size of the h5 file
N_simu_tot = 0
N_job = 0
for job_folder in job_folders:
    Path_folder = os.path.join(Path_load, job_folder)
    N_job += os.path.isdir(Path_folder)
    simu_folders = [folder for folder in os.listdir(Path_folder)]
    for simu_folder in simu_folders:
        Path_simu = os.path.join(Path_folder, simu_folder)
        N_simu_tot += os.path.isdir(Path_simu)


#%% Creation of one H5 file per job
idx_job = 0
for job_folder in job_folders:
    idx_job += 1
    print("Folder job number ", str(idx_job), " out of ", str(N_job))
    Path_h5 = os.path.join(Path_save, str(job_folder) + '_simu_100.h5')
    with h5py.File(Path_h5, "w", libver='latest') as hf:
        for name in var_names:
            data = loadmat(os.path.join(Path_ini, name + '.mat'))[name]
            size = np.shape(data)
            if name.startswith('data'):
                hf.create_dataset(name + '_r' , ((1,)+size), \
                                  maxshape=((None,)+size), \
                                  chunks=True, dtype="float64")
                hf.create_dataset(name + '_i', ((1,)+size), \
                                  maxshape=((None,)+size), \
                                  chunks=True, dtype="float64")
            elif name == 'Pos':
                hf.create_dataset("Pos", (1, 500, 4, 672), \
                                  maxshape=(None, 500, 4, 672), \
                                  chunks=True, dtype="float64")
            else:
                hf.create_dataset(name , ((1,)+size), \
                                  maxshape=((None,)+size), \
                                  chunks=True, dtype="float64")

#%% Append every simulation data in the corresponding job folder 
idx_job = 0
for job_folder in job_folders:
    idx_job += 1
    print("Folder job number ", idx_job, " out of ", str(N_job))
    Path_folder = os.path.join(Path_load, job_folder)
    simu_folders = [folder for folder in os.listdir(Path_folder) if os.path.isdir(os.path.join(Path_folder, folder))]
    idx_simu = 0
    Path_h5 = os.path.join(Path_save, str(job_folder) + '_simu_100.h5')
    with h5py.File(Path_h5, "a", libver='latest') as hf:
        for simu_folder in simu_folders:
            idx_simu += 1
            print("Simu number ", idx_simu, " out of ", str(len(simu_folders)))
            Path_simu = os.path.join(Path_folder, simu_folder)
            for name in var_names:
                
                if name.startswith('data'):
                    data_i = hf[name + '_i']
                    data_r = hf[name + '_r']
                    size = np.shape(data_i)[1:]
                    new_shape = ((idx_simu,)+size)
                    data_i.resize(new_shape)
                    data_r.resize(new_shape)
                    
                    data_mat = loadmat(os.path.join(Path_simu, name + '.mat'))[name]
                    data_i[idx_simu-1,...] = np.imag(data_mat)
                    data_r[idx_simu-1,...] = np.real(data_mat)
                elif name == 'Pos':
                    data = hf[name]
                    size = np.shape(data)[1:]
                    new_shape = ((idx_simu,)+size)
                    data.resize(new_shape)
                    
                    data_pos = loadmat(os.path.join(Path_simu, name + '.mat'))[name]
                    data[idx_simu-1,:np.shape(data_pos)[0],...] = data_pos
                else:
                    data = hf[name]
                    size = np.shape(data)[1:]
                    new_shape = ((idx_simu,)+size)
                    data.resize(new_shape)
                    
                    data[idx_simu-1,...] = loadmat(os.path.join(Path_simu, name + '.mat'))[name]
                
    
    
    
