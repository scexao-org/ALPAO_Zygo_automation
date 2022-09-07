# DM_automation

This package's main purpose is to automatize the AO3000 and zygo measurements. It can: 
- automatically take the influence function of each actuator on the DM through the zygo windows computer and send those files back to the linux DM.
- using data cubes of influence functions, reconstruct the surface of what a specific command from the DM would appear to be.
- convert .datx data into .fits data which is readable in python.

Almost all the tools to do this is created in convert_files.py. Adjust the file pathways accordingly w/in the convert_files.py class and in main.py. 
main.py initializes the convert_files class and sets up how you might want to take measurements. 

# How to configure

**Step 1**: Upload the measure.py file to the zygo computer. You can change the save path to whatever directory in the zygo computer that you desire; this will be where the data is temporarily saved.
           
           zygo1 = Zygo('C:\\Users\\zygo\\zygo_alicia\\zygo_rawdata')
           
           

**Step 2**: Upload the convert_files.py file to the linux DM computer. In the windows2linux method, change the file path to be the same as the zygo save path above. 

           file_path = 'C:/Users/zygo/zygo_alicia/zygo_rawdata/'
           
While you're there, change the ssh.exec_command to the path to your measure.py file on the zygo computer.  

           [a, b, c] = ssh.exec_command('python C:/Users/zygo/zygo_alicia/zygo_rawdata/measure.py ' + name)
           
           
           

**Step 3**: In the reset_zvals0 method, change the file_path to be the same as the zygo save path.

           file_path = 'C:/Users/zygo/zygo_alicia/zygo_rawdata/'
 
and in the ssh.exec_command 

           [a, b, c] = ssh.exec_command('python C:/Users/zygo/zygo_alicia/zygo_rawdata/measure.py ' + name)
           
and in the sftp.get, change the first string to your zygo save path. 

        sftp.get('C:\\Users\\zygo\\zygo_alicia\\zygo_rawdata\\z_init.datx', "/home/aorts/alicia/zygo_data/z_init.datx")

          


**Step 4**: Now, you need to decide where you want to save your .datx data on the linux DM computer. From now on, I'll refer to this directory as the retrieve_path. After going through data proccessing and getting rid of the piston, tip, tilt, your data will be saved to a new directory. This directory is called the save_path. In my case, my retrieve_path is "/home/aorts/alicia/zygo_rawdata/" and my save_path is "/home/aorts/alicia/zygo_data/". 

**Step 5**: Now, I'm sorry that you have to do this, but you'll need to change all the paths accordingly. I'll do my best to walk you through everything. 

In the reset_zvals0 method, change sftp.get's second string *and* the string in self._datx2py to save_path + 'z_init.datx'

        sftp.get('C:\\Users\\zygo\\zygo_alicia\\zygo_rawdata\\z_init.datx', "/home/aorts/alicia/zygo_rawdata/z_init.datx")
        h5data = self._datx2py("/home/aorts/alicia/zygo_data/z_init.datx")

In the get_zvals0 method, do the same as above. Change the string to save_path + 'z_init.datx'.

        h5data = self._datx2py('/home/aorts/alicia/zygo_data/z_init.datx')

In the analyze method, do the same as above, except this time with the self.show_h5data2 method. 

        zvals0 = self.show_h5data2('/home/aorts/alicia/zygo_data/z_init.datx')
        
Finally, in read_mult_fits_file method, change completeName accordingly to your save_path. Replace
           
        completeName = '/home/aorts/alicia/zygo_data/infl_' + str(i + 1).zfill(4) + '.fits'
        
# How to Use

Before you start using the class, there are a few house keeping things you should do. 

    dm = shm('dm64volt.im.shm') # creates connection to DM
    retrieve_path = '/home/aorts/alicia/raw_data/' # this is where your .datx data will be saved
    save_path = '/home/aorts/alicia/data/' # this is where you .fits files will be saved
    convert = con(retrieve_path, save_path) # initialize class 
    influence_functions = []
    dm_cmd = np.zeros((64, 64))

    active_actuators = convert.find_active_actuator()
    zvals0 = convert.reset_zvals0()
    nanmask = circular_aperture(0.9)(make_pupil_grid(zvals0.shape[0])).shaped
    nanmask[nanmask == 0] = np.nan

main.py is an exmaple on how I used this class. 

The most useful methods are: 
- windows2linux
- subtract
- reset_zvals0
- get_zvals0
- read_fits_file
- read_mult_fits_files

windows2linux grabs the file from zygo windows computer and saves it to the linux DM computer. Parameter 



        




           




