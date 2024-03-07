'''
Use this in the zygo windows computer.
'''

import os, sys

import os.path

import logging
import matplotlib.pyplot as plt; plt.ion()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ZYGO_LIB_PATH = "C:\ProgramData\Zygo\Mx\Scripting"
ZYGO_MEAS_SAVE_PATH = 'C:\\Users\\zygo\\zygo_alpao_feb24\\zygo_rawdata'

try:
    print("try")
    sys.path.append(ZYGO_LIB_PATH)
    from zygo import mx, instrument, connectionmanager, core
    try:
        connectionmanager.connect()
        print("try")
    except core.zygoError:
        log.warning('Zygo library loaded but connection to Mx could not be established')
except ImportError:
    log.warning('Could not load Zygo Python Library! Functionality will be severely crippled')

class Zygo:
    __save_path = ""

    def __init__(self, path):
        self.__save_path = path

    def set_path(self, path):
        self.__save_path = path

    def measurement(self, file_stem):
        instrument.measure(wait=True)
        completeName = os.path.join(self.__save_path, file_stem + ".datx")
        mx.save_data(completeName)
        log.warning(f"Measurement taken to {file_stem}.datx")

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
    zygo1 = Zygo(ZYGO_MEAS_SAVE_PATH)
    zygo1.measurement(filename)
    print(filename + ' Done!')