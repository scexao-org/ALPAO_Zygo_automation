'''
Run this to take a data cube. 
'''

from hcipy import *
import numpy as np
from tqdm import tqdm
from pyMilk.interfacing.shm import SHM as shm
from convert_files import ConvertFiles

'''
MAIN PATHS
'''
LNX_RAW_DATA = '/home/scexao/alpao/raw_data/'
LNX_DATA = '/home/scexao/alpao/data/'
WIN_RAWDATA = 'C:/Users/zygo/zygo_alpao_feb24/zygo_rawdata/'
WIN_RAWDATA_BACKWARDS = WIN_RAWDATA.replace('/', '\\')

if __name__ == '__main__':

    dm = shm('dm64in')
    convert = ConvertFiles(LNX_RAW_DATA, LNX_DATA)
    convert.get_zvals0()
   
    influence_functions = []
    dm_cmd = np.zeros((64, 64))

    active_actuators = convert.find_active_actuator()
    zvals0 = convert.reset_zvals0()
    nanmask = circular_aperture(0.9)(make_pupil_grid(zvals0.shape[0])).shaped
    nanmask[nanmask == 0] = np.nan

    # for i in tqdm(range(active_actuators.shape[0]))
    # for i in tqdm((active_actuators.shape[0]), 0, -1):
    for i in tqdm(range(1925, 0, -1)):
        # pos
        amplitude = 0.5
        dm_cmd_pos = np.reshape(active_actuators[i-1], [64, 64]) * amplitude
        dm.set_data(dm_cmd_pos.astype(np.float32))
        pos_name = 'infl_' + str(i).zfill(4) + '_pos.datx'
        convert.windows2linux(pos_name)
        # neg
        dm_cmd_neg = np.reshape(active_actuators[i-1], [64, 64]) * -1 * amplitude
        dm.set_data(dm_cmd_neg.astype(np.float32))
        neg_name = 'infl_' + str(i).zfill(4) + '_neg.datx'
        convert.windows2linux(neg_name)

        # subtract
        diff = convert.subtract(pos_name, neg_name)
        diff = diff * (632.8/2.0)
        influence_functions.append(diff)
