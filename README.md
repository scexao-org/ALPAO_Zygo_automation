# DM_automation

This package's main purpose is to automatize the AO3000 and zygo measurements. It can: 
- automatically take the influence function of each actuator on the DM through the zygo windows computer and send those files back to the linux DM.
- using data cubes of influence functions, reconstruct the surface of what a specific command from the DM would appear to be.
- convert .datx data into .fits data which is readable in python.

Almost all the tools to do this is created in convert_files.py. Adjust the file pathways accordingly w/in the convert_files.py class and in main.py. 
main.py initializes the convert_files class and sets up how you might want to take measurements. 

Step 1: Upload the measure.py file to the zygo computer. You can change the save path to whatever directory you desire
           zygo1 = Zygo(**'C:\\Users\\zygo\\zygo_alicia\\zygo_rawdata'**)

Step 2: 
