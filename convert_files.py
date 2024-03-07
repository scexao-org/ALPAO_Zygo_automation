import h5py
import paramiko
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from hcipy import *
from astropy.io import fits


from pyMilk.interfacing.shm import SHM

dm_shm = SHM('dm64in')

def connect_paramiko():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='133.40.145.223', username='Administrator', password=b'zygo22')
    return ssh

# set ssh
ssh = connect_paramiko()
sftp = ssh.open_sftp()

LNX_RAW_DATA = '/home/scexao/alpao/raw_data'
LNX_DATA = '/home/scexao/alpao/data'
WIN_RAWDATA = 'C:/Users/zygo/zygo_alpao_feb24/zygo_rawdata'
WIN_RAWDATA_BACKWARDS = WIN_RAWDATA.replace('/', '\\')

ZYGO_SCRIPT = 'C:/Users/zygo/python/measure.py'

class ConvertFiles:
    _save_path = ""
    _retrieve_path = ""
    
    def __init__(self, retrieve_path, save_path, use_shm: bool = False, dm_obj = None):
        # retrieve path is where .datx files are saved.
        # save path is where post-data processing .fits files are saved. 
        self._retrieve_path = retrieve_path
        self._save_path = save_path
        self.use_shm = use_shm
        self.dm_obj = dm_obj

    def set_data(self, data):
        if self.use_shm:
            dm_shm.set_data(data)
        else:
            self.dm_obj.set_command_64x64(data)

        
    
    def find_active_actuator(self, nact=64, ca=1):
        # creates a mask to iterate through the active actuators. 
        pupil_grid = make_pupil_grid(nact)
        mask = circular_aperture(ca)(pupil_grid)
        idx = np.where(mask == 1)[0]
        actuators = np.eye(nact * nact)
        active_actuator = actuators[idx]
        return active_actuator

    def windows2linux(self, file_stem):
        # transfer file from zygo windows computer to dm linux computer.

        name = WIN_RAWDATA + '/' + file_stem + '.datx'
        name_bs = WIN_RAWDATA_BACKWARDS + '\\' + file_stem + '.datx'

        file = self._retrieve_path + '/' + file_stem + '.datx'
        [a, b, c] = ssh.exec_command(
            f'python {ZYGO_SCRIPT} {file_stem}')
        print(b.read())
        sftp.get(name_bs, file)

    def push_dm(self, dmx, dmy, poke=0.1):
        # ask the dm to push a specific actuator in x, y coordinates. 
        x = np.zeros(((64, 64)), dtype=np.float64)
        x[dmx][dmy] = poke
        self.set_data(x)
        name = f'{dmx}_{dmy}_{poke}poke.datx'
        self.windows2linux(name)
        print("measurement has been taken and saved to " + name)
        x[dmx][dmy] = 0
        self.set_data(x)

    def push_dm_multi(self,dmx_list,dmy_list, poke=0.1):
        #ask the dm to push multiple actuators as defined by list of x,y coordinates
        x = np.zeros(((64, 64)), dtype=np.float64)
        for i in dmx_list:
            for j in dmy_list:
                x[i][j] = poke
        print(x)
        self.set_data(x)
        name = f'{dmx_list[0]}_{dmy_list[0]}_{poke}poke3x3.datx'
        self.windows2linux(name)
        print("measurement has been taken and saved to " + name)
        for i in dmx_list:
            for j in dmy_list:
                x[i][j] = 0.0
        self.set_data(x)

    def show_h5data(self, file_name):
        # show the data of a .datx files. Return its data in the form of a numpy array.
        h5data = _datx2py(self._retrieve_path + file_name)
        zdata = h5data['Data']['Surface']
        zdata = list(zdata.values())[0]
        zvals = zdata['vals']
        zvals[zvals == zdata['attrs']['No Data']] = np.nan
        zunit = zdata['attrs']['Z Converter']['BaseUnit']
        pv = np.nanmax(zvals) - np.nanmin(zvals)
        rms = np.sqrt(np.nanmean((zvals - np.nanmean(zvals)) ** 2))
        np.nan_to_num(zvals, copy=False)
        plt.pcolormesh(zvals)
        plt.title("{0:0.3f} {2} PV, {1:0.3f} {2} RMS".format(pv, rms, zunit))
        plt.show()
        return zvals

    def show_h5data2(self, file_path):
        # same as above but the method variable is in the form of a file path instead of a file name. 
        h5data = _datx2py(file_path)
        zdata = h5data['Data']['Surface']
        zdata = list(zdata.values())[0]
        zvals = zdata['vals']
        zvals[zvals == zdata['attrs']['No Data']] = np.nan
        zunit = zdata['attrs']['Z Converter']['BaseUnit']
        pv = np.nanmax(zvals) - np.nanmin(zvals)
        rms = np.sqrt(np.nanmean((zvals - np.nanmean(zvals)) ** 2))
        np.nan_to_num(zvals, copy=False)
        plt.pcolormesh(zvals)
        plt.title("{0:0.3f} {2} PV, {1:0.3f} {2} RMS".format(pv, rms, zunit))
        plt.show()
        return zvals

    def reset_zvals0(self):
        # Set dm to 0 and reset zvals. 
        x = np.zeros((64, 64), dtype=np.float32)
        self.set_data(x)
        ssh = connect_paramiko()
        name = WIN_RAWDATA + '/z_init.datx'

        [a, b, c] = ssh.exec_command(f'python {ZYGO_SCRIPT} {name}')
        print(b.read())
        sftp = ssh.open_sftp()
        sftp.get(WIN_RAWDATA_BACKWARDS + "\\z_init.datx",
                 self._retrieve_path + "/z_init.datx")
        h5data = _datx2py(self._retrieve_path + "/z_init.datx")

        zdata = h5data['Data']['Surface']
        zdata = list(zdata.values())[0]
        zvals = zdata['vals']
        zvals[zvals == zdata['attrs']['No Data']] = np.nan
        np.nan_to_num(zvals, copy=False)
        return zvals

    def get_zvals0(self):
        # return zvals 
        h5data = _datx2py(self._save_path + "/z_init.datx")
        zdata = h5data['Data']['Surface']
        zdata = list(zdata.values())[0]
        zvals = zdata['vals']
        zvals[zvals == zdata['attrs']['No Data']] = np.nan
        np.nan_to_num(zvals, copy=False)
        return zvals

    def analyze(self, dmx, dmy, amplitude):
        # will take a measurement of a specific actuator w/ a specific amplitude. 
        x = np.zeros((64, 64), dtype=np.float32)
        x[dmx][dmy] = amplitude
        self.set_data(x)
        name = str(dmx) + "_" + str(dmy) + "_" + str(amplitude) + ".datx"
        self.windows2linux(name)
        print("measurement has been taken and saved to " + name)
        x[dmx][dmy] = 0
        zvals = self.show_h5data(name)
        zvals0 = self.show_h5data2(self._save_path + '/z_init.datx')
        nanmask = circular_aperture(0.9)(make_pupil_grid(zvals0.shape[0])).shaped
        nanmask[nanmask == 0] = np.nan
        diff = zvals - zvals0
        diff = diff #* (632.8 / 2.0)
        diff_wo_ptt = self.remove_ptt(diff, ca=0.9)
        plt.figure(1)
        plt.subplot(1, 3, 1)
        plt.imshow(diff * nanmask, cmap='jet')
        plt.title('Influence function w PTT')
        plt.gca().invert_yaxis()
        plt.subplot(t1, 3, 2)
        plt.imshow(diff_wo_ptt * nanmask, cmap='jet')
        plt.title('Influence function wo PTT')
        plt.gca().invert_yaxis()
        plt.show()
        new_filename = name[slice(0, -5)]
        completeName = self._save_path + new_filename + ".fits"
        write_fits(diff_wo_ptt.ravel(), completeName)
        print(".datx file difference has been made and was saved to " + completeName)
        self.read_fits_file(completeName)
        return diff_wo_ptt

    def analyze_data(self, file_zvalsa):
        # takes .datx file and data process and save as .fits file. 
        h5data = _datx2py(self._retrieve_path + '/' + file_zvalsa)
        zdata = h5data['Data']['Surface']
        zdata = list(zdata.values())[0]
        zvalsa = zdata['vals']
        zvalsa[zvalsa == zdata['attrs']['No Data']] = np.nan
        np.nan_to_num(zvalsa, copy=False)
        try:
            zvals0 = self.get_zvals0()
        except OSError:
            print('NO z_init.datx file for now.')
            zvals0 = np.zeros_like(zvalsa)
        difference = zvalsa - zvals0
        difference = self.remove_ptt(difference)
        new_filename = file_zvalsa[slice(0, -5)]
        difference = difference #* (632.8 / 2)
        complete_name = self._save_path + new_filename + ".fits"
        write_fits(difference.ravel(), complete_name)
        print(".datx difference has been made and was saved to " + complete_name)
        return difference

    def subtract(self, file_zvals_pos, file_zvals_neg, poke):
        # takes the difference between two influence functions. Saves as .fits file.
        pos = _datx2py(self._retrieve_path + file_zvals_pos)
        pos_zdata = pos['Data']['Surface']
        pos_zdata = list(pos_zdata.values())[0]
        pos_zvals = pos_zdata['vals']
        pos_zvals[pos_zvals == pos_zdata['attrs']['No Data']] = np.nan
        np.nan_to_num(pos_zvals, copy=False)

        neg = _datx2py(self._retrieve_path + file_zvals_neg)
        neg_zdata = neg['Data']['Surface']
        neg_zdata = list(neg_zdata.values())[0]
        neg_zvals = neg_zdata['vals']
        neg_zvals[neg_zvals == neg_zdata['attrs']['No Data']] = np.nan
        np.nan_to_num(neg_zvals, copy=False)

        dif = pos_zvals - neg_zvals
        dif = self.remove_ptt(dif)
        new_filename = file_zvals_neg[slice(0, -9)]
        difference = dif * (1/(2.0*poke))   #* (632.8 / 2.0)
        complete_name = self._save_path + new_filename + '.fits'
        write_fits(difference.ravel(), complete_name)
        print(".datx difference has been made and file was saved to " + complete_name)
        return difference

    def read_fits_file(self, file_path):
        # will read a single .fits file.
        nanmask = circular_aperture(0.9)(make_pupil_grid(self.get_zvals0().shape[0])).shaped
        nanmask[nanmask == 0] = np.nan
        file_name = file_path[-14:]
        image_file = file_path
        image_data = fits.getdata(image_file, ext=0)
        plt.figure()
        plt.imshow(image_data * nanmask, cmap='jet', vmin=-800, vmax=200)
        plt.title(file_name)
        plt.colorbar(label='nm')
        plt.gca().invert_yaxis()

    def read_mult_fits_file(self, start, stop):
        # will read multiple fits files. 
        influence_functions = []
        for i in range(start, stop):
            completeName = self._save_path + '/infl_' + str(i + 1).zfill(4) + '.fits'
            nanmask = circular_aperture(0.9)(make_pupil_grid(self.get_zvals0().shape[0])).shaped
            nanmask[nanmask == 0] = np.nan
            image_data = fits.getdata(completeName, ext=0)
            influence_functions.append(image_data)

        fig = plt.figure(1)
        for i in range(len(influence_functions)):
            inf = influence_functions[i]
            plt.clf()
            plt.figure(1)
            plt.imshow(inf * nanmask, cmap='jet', vmin=-100, vmax=100)
            plt.colorbar(label='nm')
            plt.title('infl' + str(start + 1).zfill(4))
            plt.gca().invert_yaxis()
            fig.canvas.draw()
            fig.canvas.flush_events()
            start += 1
            plt.show()

    def remove_ptt(self, data, ca=0.9):
        # removes piston, tip, and tilt. 
        num_pix = data.shape[0]
        num_modes = 3  # number of zernike modes
        mask = circular_aperture(ca)(make_pupil_grid(num_pix, 1)).shaped
        data = data * mask

        # Takes a long time (put in init)
        z_basis = make_zernike_basis(num_modes, ca, make_pupil_grid(num_pix, diameter=1))  # zernike basis
        # Substract Piston, Tip, Tilt
        piston_tip_tilt = z_basis[0:3]  # Piston, Tip, Tilt = z_basis[0], z_basis[1], z_basis[2]
        c_ptt = piston_tip_tilt.coefficients_for(data.ravel())  # coefficients only for Piston, Tip, Tilt
        ptt_recon = piston_tip_tilt.linear_combination(c_ptt).shaped  # "linear combination" is useful a function
        data_wo_ptt = data - ptt_recon  # substract piston, tip, tilt from the original data
        return data_wo_ptt

import pathlib

def convert_datx_fits(file_fullpath, target_folder,overwrite=True):
    h5data = list(_datx2py(file_fullpath)['Data']['Surface'].values())[0]
    h5vals = h5data['vals']
    h5vals[h5vals == h5data['attrs']['No Data']] = 0.0

    p_datx = pathlib.Path(file_fullpath)
    fitspath = target_folder + '/' + p_datx.stem + '.fits'

    if overwrite == True:
        fits.writeto(fitspath, h5vals, overwrite=True)
    else:
        fits.writeto(fitspath, h5vals)


def _datx2py(file_name):
    # unpacks ,datx files into h5data. 
    # unpack an h5 group into a dict
    def _group2dict(obj):
        return {k: _decode_h5(v) for k, v in zip(obj.keys(), obj.values())}

    # unpack a numpy structured array into a dict
    def _struct2dict(obj):
        names = obj.dtype.names
        return [dict(zip(names, _decode_h5(record))) for record in obj]

    # decode h5py.File object and all of its elements recursively
    def _decode_h5(obj):
        # group -> dict
        if isinstance(obj, h5py.Group):
            d = _group2dict(obj)
            if len(obj.attrs):
                d['attrs'] = _decode_h5(obj.attrs)
            return d
        # attributes -> dict
        elif isinstance(obj, h5py.AttributeManager):
            return _group2dict(obj)
        # dataset -> numpy array if not empty
        elif isinstance(obj, h5py.Dataset):
            d = {'attrs': _decode_h5(obj.attrs)}
            try:
                d['vals'] = obj[()]
            except (OSError, TypeError):
                pass
            return d
        # numpy array -> unpack if possible
        elif isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.number) and obj.shape == (1,):
                return obj[0]
            elif obj.dtype == 'object':
                return _decode_h5([_decode_h5(o) for o in obj])
            elif np.issubdtype(obj.dtype, np.void):
                return _decode_h5(_struct2dict(obj))
            else:
                return obj
        # dimension converter -> dict
        elif isinstance(obj, np.void):
            return _decode_h5([_decode_h5(o) for o in obj])
        # bytes -> str
        elif isinstance(obj, bytes):
            return obj.decode()
        # collection -> unpack if length is 1
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 1:
                return obj[0]
            else:
                return obj
        # other stuff
        else:
            return obj

    # open the file and decode it
    with h5py.File(file_name, 'r') as f:
        h5data = _decode_h5(f)

    return h5data
