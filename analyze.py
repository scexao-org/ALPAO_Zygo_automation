from __future__ import annotations
import typing as typ
if typ.TYPE_CHECKING:
    DatDict = typ.Dict[typ.Tuple[int, int, int], np.ndarray]

import pathlib
import glob
import numpy as np
from astropy.io import fits


import matplotlib.pyplot as plt; plt.ion()

DATAPATH = '/home/scexao/alpao/data'
DATAPATH_POW = '/home/scexao/alpao/power'

tiparr = np.arange(1200) - 1199/2.
TIP_MAP = tiparr[None,:] * np.ones((1200, 1))
TILT_MAP = tiparr[:,None] * np.ones((1, 1200))
TIP_MAP /= np.std(TIP_MAP)
TILT_MAP /= np.std(TILT_MAP)


def parse_name(file_name: str) -> tuple[int, int, int]:
    stem = pathlib.Path(file_name).stem
    _, i, j, p = stem.split('_')

    return int(i), int(j), int(p)

def load_files(file_list: list[str]) -> DatDict:
    all_ret: DatDict = {}
    for file in file_list:
        arr = fits.getdata(file)
        arr[arr == 0.0] = np.nan
        all_ret[parse_name(file)] = arr

    return all_ret

def compute_ref(dat: DatDict) -> np.ndarray:
    pm25s = [d for (i,j,p), d in dat.items() if abs(p) == 25]

    return sum(pm25s) / len(pm25s)

def detilt(img: np.ndarray, tip: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    n_px = np.sum(~np.isnan(img))
    scale = n_px / img.size
    p_tip = np.nanmean(img * tip) / scale
    p_tilt = np.nanmean(img * tilt) / scale

    return img - p_tip*tip - p_tilt*tilt - np.nanmean(img)


def plot_pokeseries6(i: int, j: int, dat: DatDict, ref: np.ndarray):
    relevant_data = {p: d for (ii, jj, p), d in dat.items() if (ii, jj) == (i, j)}
    p_vals = np.array(list(relevant_data.keys()))
    p_vals.sort()

    fig = plt.figure(6)
    fig.clf()
    axs = fig.subplots(2, 3)

    for k, ax in enumerate(axs.flat):
        mat = ax.matshow(detilt(relevant_data[p_vals[k]] - ref, TIP_MAP, TILT_MAP), vmin=-3.0, vmax=3.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(f'Actu {i} {j} - stroke {p_vals[k]}')

    plt.colorbar(mat)
    plt.tight_layout()

def plot_linlines_allactu_coarse(dat, ref, inv_value = 0):
    fig = plt.figure(8)
    fig.clf()
    ax = fig.subplots()
    for i in range(51, 62):
        for j in range(27, 38):
            relevant_data = {p: d for (ii, jj, p), d in dat.items() if (ii, jj) == (i, j)}
            if len(relevant_data) == 0:
                continue
            p_vals = np.array(list(relevant_data.keys()))
            p_vals.sort()

            p2v = np.zeros(p_vals.shape)
            for k, p in enumerate(p_vals):
                map = detilt(dat[(i,j,p)] - ref, TIP_MAP, TILT_MAP)
                #sign = (np.nanmean(map[350:850, 90:450]) > 0) * 2.0 - 1.0
                p2v[k] = (np.nanmax(map) - np.nanmin(map)) * -np.sign(p - inv_value)
            if i == 56 and j == 32:
                ax.plot(p_vals, p2v, 'r-', lw=3)
            else:
                ax.plot(p_vals, p2v, 'k-', lw=1)

    ax.set_xlabel('Stroke (%)')
    ax.set_ylabel('Displacement (um WF)')
    ax.set_title('Linearity curves for 11x11 actuators')

def lindata_plot_poked_area(dat):
    by_p = {}
    for (i,j,p), data in dat.items(): 
        d = by_p.get(p, {}) 
        d[(i,j)] = data 
        by_p[p] = d

    n = len(by_p[100])
    a = sum(by_p[100].values())
    b = sum(by_p[-100].values())

    fig = plt.figure(21)
    fig.clf()
    ax = fig.subplots()
    mat = ax.matshow(b-a)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.colorbar(mat)


def plot_power(pow_temp_data):
    # /home/scexao/alpao/power/power_test.fits: 2x4x64x64
    pow_data = pow_temp_data[:, 2:].reshape(2, 2, 4096)

    push = pow_data[0]
    pull = pow_data[1]

    fig = plt.figure(7)
    fig.clf()
    axs = fig.subplots(8, 1)

    from hwmain.alpao64 import lut
    binLUT = lut.LUT_4096_TO_3228 > 0
    
    for kk, ax in enumerate(axs.flat):
        sls = np.s_[kk*512:(kk+1)*512]
        ax.plot(push[:,sls].T, color='C0')
        ax.plot(pull[:,sls].T, color='C1')

        ax.plot((push-pull)[0][sls], color='C3')


        ax.plot(binLUT[sls] / 20., 'k')
        ax.set_ylim(-0.25, 0.25)
        ax.text(0, 0.2, f'Rack {kk} - 512 channels')

    assert np.all((push-pull)[0][binLUT]) > 0.05

    axs[0].set_xlabel('Channel number')
    axs[0].set_ylabel('Measured amps')
    axs[0].set_title('Amp measured for single active channels (push-pull 30% of stroke)\n'
                 'Blue/Orange: push-pull for the 2 amp values retrieved; Red: difference; black: DM channels')
    
    fig = plt.figure(8)
    fig.clf()
    ax = fig.subplots()

    mshow = ax.matshow((push-pull)[0][lut.LUT_64x64_TO_4096].reshape(64,64) * lut.CROP_USEFUL_64x64_TO_3228)
    plt.colorbar(mshow)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Amp difference for -30%/+30% stroke - actu map')

def plot_ref(ref):
    fig = plt.figure(3)
    fig.clf()
    plt.matshow(ref - np.nanmean(ref, ))
    plt.title('Ref. poweroff wavefront')
    plt.colorbar()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_pokes():
    pass

if __name__ == "__main__":

    #poke_files = glob.glob(DATAPATH + "/poke_*.fits")
    poke_files = glob.glob(DATAPATH + "/pokeswap_*.fits")
    #pokedetail_files = glob.glob(DATAPATH + "/pokedetail_*.fits")


    dat = load_files(poke_files)
    #datdetail = load_files(pokedetail_files)
    ref = compute_ref(dat)

    tip_files = glob.glob(DATAPATH + "/tip_*.fits")
    tilt_files = glob.glob(DATAPATH + "/tilt_*.fits")

    tip_dat = load_files(tip_files)
    tilt_dat = load_files(tilt_files)


