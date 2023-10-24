#%% Initialization
from scipy.io import loadmat 
import h5py
import numpy as np
import os
import time


Path_save = r'C:\Users\Asus\Documents\3A\Montreal\Stage\code\data\h5'
Path_load = r'C:\Users\Asus\Documents\3A\Montreal\Stage\code\data\small_train_data'
Path_ini  = os.path.join(Path_load, str(1), str(1))

var_names = [file[:-4] for file in os.listdir(Path_ini)]

# Help to initialize the size of the h5 file
N_simu = 9
N_job = 2

#%% Creation of one H5 file per job
for idx_job in range(1, 3):
    print("Folder job number ", str(idx_job), " out of ", str(N_job))
    Path_folder = os.path.join(Path_save, str(idx_job))
    os.mkdir(os.path.join(Path_save, str(idx_job)))
    Path_h5 = os.path.join(Path_save, str(idx_job), 'simu_100.h5')
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
for idx_job in range(1, 3):
    print("Folder job number ", str(idx_job), " out of ", str(N_job))
    Path_folder = os.path.join(Path_load, str(idx_job))
    Path_h5 = os.path.join(Path_save, str(idx_job), 'simu_100.h5')               
    for idx_simu in range(1, N_simu+1):
        print("Simu number ", str(idx_simu), " out of ", str(N_simu))
        Path_simu = os.path.join(Path_folder, str(idx_simu))
        with h5py.File(Path_h5, "a", libver='latest') as hf:
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
                    
with h5py.File(Path_h5, 'r') as hdf_file:
    # Print the keys at the root level of the HDF5 file
    print("Keys in the HDF5 file:", list(hdf_file.keys()))
                
    
    
    
