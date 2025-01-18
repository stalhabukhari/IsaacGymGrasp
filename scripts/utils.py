import numpy as np
import h5py


def write_hdf5_data(filename, data):
    with h5py.File(filename, "w") as hf:
        for key, value in data.items():
            hf.create_dataset(key, data=value)


def read_hdf5_data(filename):
    data = {}
    with h5py.File(filename, "r") as hf:
        for key in hf.keys():
            data[key] = np.array(hf[key])
    return data
