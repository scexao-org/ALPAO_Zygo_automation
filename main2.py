import os, sys
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import paramiko
from pyMilk.interfacing.shm import SHM as shm
from convert_files import ConvertFiles as ConvertFiles

if __name__ == '__main__':

    dm = shm('dm64volt.im.shm')
    retrieve_path = '/home/aorts/alicia/zygo_data/'
    save_path = '/home/aorts/alicia/data/'
    convert = ConvertFiles(retrieve_path, save_path)
    convert.get_zvals0()
    file_path = 'C:/Users/zygo/zygo_alicia/zygo_rawdata/'

    influence_functions = []
    dm_cmd = np.zeros((64, 64))

    active_actuators = convert.find_active_actuator()
    zvals0 = convert.reset_zvals0()
    nanmask = circular_aperture(0.9)(make_pupil_grid(zvals0.shape[0])).shaped
    nanmask[nanmask == 0] = np.nan

    # for i in tqdm(range(active_actuators.shape[0]))
    for i in tqdm(range(30)):
        # pos
        amplitude = 0.5
        dm_cmd_pos = np.reshape(active_actuators[i], [64, 64]) * 0.5
        dm.set_data(dm_cmd_pos.astype(np.float32))
        pos_name = 'infl_' + str(i + 1).zfill(4) + '_pos.datx'
        convert.windows2linux(pos_name)
        # neg
        dm_cmd_neg = np.reshape(active_actuators[i], [64, 64]) * -0.5
        dm.set_data(dm_cmd_neg.astype(np.float32))
        neg_name = 'infl_' + str(i + 1).zfill(4) + '_neg.datx'
        convert.windows2linux(neg_name)

        # subtract
        diff = convert.subtract(pos_name, neg_name)
        diff = diff * (632.8/2.0)
        influence_functions.append(diff)

    # Shows plots through influence functions
    # dm.set_data(np.zeros((64, 64)).astype(np.float32))
    # for i in range(len(influence_functions)):
    #     inf = influence_functions[i]
    #     plt.clf()
    #     plt.figure(1)
    #     plt.gca().invert_yaxis()
    #     plt.imshow(inf * nanmask, cmap='jet')
    #     plt.title('infl' + str(i + 1).zfill(4))
    #     plt.show()

    convert.read_mult_fits_file(start=0, stop=30)
