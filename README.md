# DM_automation

This package's main purpose is to automatize the AO3000 and zygo measurements. It can: 
- take the influence function of each actuator on the DM through the zygo windows computer and send those files back to the linux DM
- using data cubes, reconstruct the surface of what a specific command from the DM would appear to be.

Almost all the tools to do this is created in convert_files.py. Adjust the file pathways accordingly w/in the convert_files.py class and in main.py. 
main.py initializes the convert_files class and sets up how you might want to take measurements. 
