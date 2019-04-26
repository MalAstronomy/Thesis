import numpy as np
import h5py 
import os
import shutil
from os import path
import json
import glob

f= h5py.File('/disks/strw9/vasist/MasterThesis2/mergers_identified/mergers_28.hdf5', 'r')

mass_ratio=f.get('Mass Ratio').value
####################################################################################################################
os.mkdir('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.16_0.25/')
params={}
params["merger_settings"]=[]
params["merger_settings"].append({
	"mass_ratio": 0.250000
})
with open('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.16_0.25/params.json', 'w') as outfile:  
    json.dump(params, outfile)
####################################################################################################################
os.mkdir('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.25_0.5/')
params={}
params["merger_settings"]=[]
params["merger_settings"].append({
	"mass_ratio": 0.50000
})
with open('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.25_0.5/params.json', 'w') as outfile:  
    json.dump(params, outfile)
####################################################################################################################
os.mkdir('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.5_0.75/')
params={}
params["merger_settings"]=[]
params["merger_settings"].append({
	"mass_ratio": 0.750000
})
with open('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.5_0.75/params.json', 'w') as outfile:  
    json.dump(params, outfile)
####################################################################################################################
os.mkdir('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.75_1.0/')
params={}
params["merger_settings"]=[]
params["merger_settings"].append({
	"mass_ratio": 1.0000
})
with open('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.75_1.0/params.json', 'w') as outfile:  
    json.dump(params, outfile)
####################################################################################################################
for i in range(len(mass_ratio)): 
	if i == 37:continue
	if i == 89:continue
	if i == 95:continue
	if i == 109:continue
	if i == 117:continue
	if i == 123:continue
	if i == 124:continue
	if i == 125:continue
	
        if mass_ratio[i] > 0.16 and mass_ratio[i] < 0.25:
                for angle in range(0,360,15):
                    src = path.realpath('/disks/strw9/vasist/MasterThesis2/data_visualisation/images_28/images_rotated/28_' + str(i) + '_'+ str(angle) + '.jpg')
		    dst=  path.realpath('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.16_0.25/')		
                    shutil.copy(src,dst)

	elif mass_ratio[i] > 0.25 and mass_ratio[i] < 0.5:
		for angle in range(0,360,15):
                    src = path.realpath('/disks/strw9/vasist/MasterThesis2/data_visualisation/images_28/images_rotated/28_' + str(i) + '_'+ str(angle) + '.jpg')
                    dst=  path.realpath('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.25_0.5/') 
                    shutil.copy(src,dst)

	elif mass_ratio[i] > 0.5 and mass_ratio[i] < 0.75:
	        for angle in range(0,360,15):
                    src = path.realpath('/disks/strw9/vasist/MasterThesis2/data_visualisation/images_28/images_rotated/28_' + str(i) + '_'+ str(angle) + '.jpg')
                    dst=  path.realpath('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.5_0.75/') 
                    shutil.copy(src,dst)

	elif mass_ratio[i] > 0.75 and mass_ratio[i] < 1.0:
                for angle in range(0,360,15):
                    src = path.realpath('/disks/strw9/vasist/MasterThesis2/data_visualisation/images_28/images_rotated/28_' + str(i) + '_'+ str(angle) + '.jpg')
                    dst=  path.realpath('/disks/strw9/vasist/MasterThesis2/data_classes_rotated/mr_0.75_1.0/') 
                    shutil.copy(src,dst)
