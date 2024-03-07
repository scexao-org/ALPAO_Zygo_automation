import numpy as np
from convert_files import *
from tqdm import tqdm

from hwmain.alpao64.blue import dm

convert = ConvertFiles(LNX_RAW_DATA, LNX_DATA, dm_obj = dm) #data from dm, data after convert, shared memory object

act_count = np.arange(0,62,1)

for i,x in tqdm(enumerate(act_count)):
    for j,y in enumerate(act_count):
        dmx_list = [x,x+1,x+2]
        dmy_list = [y,y+1,y+2]
        poke = 0.1
        convert.push_dm_multi(dmx_list,dmy_list,poke)
        name = f'{dmx_list[0]}_{dmy_list[0]}_{poke}poke3x3.datx'
        convert_datx_fits(LNX_RAW_DATA+f'/{name}.datx', LNX_DATA)