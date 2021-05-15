# import scipy.io
# mat = scipy.io.loadmat('/home/slam_data/data_sets/validation-set.mat')
from pprint import pprint
import numpy as np
import h5py
with h5py.File('/home/slam_data/data_sets/validation-set.mat', 'r') as f:
    print(f.keys())
    print(f['data'][0, 0])
    # pprint(dir(f['data'][0, 0]))

    # ref = f['labels'][0, 20]

    ref = f['#refs#']
    print(ref)
    exit()
    print(np.array(f[ref]))
    exit()