# DM_automation

This package's main purpose is to automatize the AO3000 and zygo measurements. It can: 
- automatically take the influence function of each actuator on the DM through the zygo windows computer and send those files back to the linux DM.
- using data cubes of influence functions, reconstruct the surface of what a specific command from the DM would appear to be.
- convert .datx data into .fits data which is readable in python.

Almost all the tools to do this is created in convert_files.py. Adjust the file pathways accordingly w/in the convert_files.py class and in main.py. 
main.py initializes the convert_files class and sets up how you might want to take measurements. 

Step 1: Upload the measure.py file to the zygo computer. You can change the save path to whatever directory in the zygo computer that you desire; this will be where the data is temporarily saved.
           
           zygo1 = Zygo('C:\\Users\\zygo\\zygo_alicia\\zygo_rawdata')

Step 2: Upload the convert_files.py file to the linux DM computer. In the windows2linux method, change the file path to be the same as the zygo save path above. 

           file_path = 'C:/Users/zygo/zygo_alicia/zygo_rawdata/'
           
While you're there, change the ssh.exec_command to the path to your measure.py file on the zygo computer.  

           [a, b, c] = ssh.exec_command('python C:/Users/zygo/zygo_alicia/zygo_rawdata/measure.py ' + name)

Step 3: In the reset_zvals0 method, change the file_path to be the same as the zygo save path.

           file_path = 'C:/Users/zygo/zygo_alicia/zygo_rawdata/'
 
and in the ssh.exec_command 

           [a, b, c] = ssh.exec_command('python C:/Users/zygo/zygo_alicia/zygo_rawdata/measure.py ' + name)

Step 4: Now, you need to decide where you want to save your .datx data on the linux DM computer. From now on, I'll refer to this directory as the retrieve_path. After going through data proccessing and getting rid of the piston, tip, tilt, your data will be saved to a new directory. This directory is called the save_path. In my case, my retrieve_path is "/home/aorts/alicia/zygo_data/" and my save_path is "/home/aorts/alicia/data/". 




           




