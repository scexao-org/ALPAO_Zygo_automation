from hcipy import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt; plt.ion()

mode_basis = []
for i in tqdm(range(3228)):
    mode = read_fits('/home/aorts/alicia/data/infl_' + str(i+1).zfill(4) + '.fits')
    mode_basis.append(mode.ravel())
    
mode_basis = ModeBasis(mode_basis)
# mode_basis.grid = make_pupil_grid(1200) # to save the mode basis, we need pupil_grid
# write_mode_basis(mode_basis, '/home/aorts/kyohoon/alpao_mode_basis.fits') # something wrong, I will figure it out on Monday

# reconstruct map
dm_mask = circular_aperture(1)(make_pupil_grid(64))
idx = np.where(dm_mask==1)[0]
dm_cmd = np.zeros((64, 64))
dm_cmd[10:15, :] = 1

recon_map = mode_basis.linear_combination(dm_cmd.ravel()[idx])
recon_map = np.reshape(recon_map, [1200, 1200])
plt.imshow(recon_map, cmap='jet', origin='bottom')
