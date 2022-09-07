'''
Use this in the zygo windows computer.
'''

import os, sys
import numpy as np
import h5py
import os.path
import logging
import matplotlib.pyplot as plt; plt.ion()
from hcipy import *
import utils
from scipy import interpolate
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

try:
    print("try")
    sys.path.append('C:\ProgramData\Zygo\Mx\Scripting')
    import zygo
    from zygo import mx, instrument, systemcommands, connectionmanager, ui, core
    from zygo.units import Units
    try:
        connectionmanager.connect()
        print("try")
    except core.zygoError:
        log.warning('Zygo library loaded but connection to Mx could not be established')
except ImportError:
    log.warning('Could not load Zygo Python Library! Functionality will be severely crippled')

class Zygo:

    __filename = None
    __save_path = ""

    def __init__(self, filename, path):
        self.__filename = filename
        self.__save_path = path

    def set_path(self, path):
        self.__save_path = path

    def actuator_measurement(self, x_actuators, y_actuators):
        for x in range(x_actuators):
            for y in range(y_actuators):
                instrument.measure(wait=True)
                completeName = os.path.join(self.__save_path, str(x) + "_" + str(y) + ".datx")
                mx.save_data(completeName)
                log.warning("Measurement has been taken and saved for movement of actuator " + str(x) + ", " + str(y))
                # something here that will wait until the dm has moved another actuator

if __name__ == "__main__":
    filename = sys.argv[1]
    zygo1 = Zygo('C:\\Users\\zygo\\zygo_alicia\\zygo_rawdata')
    zygo1.measurement(filename)
    print(filename + ' Done!')