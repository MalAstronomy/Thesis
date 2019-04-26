import numpy as np
import h5py 
import os
import shutil
from os import path
import json
import glob

f= h5py.File('/disks/strw9/vasist/MasterThesis2/mergers_identified/mergers_28.hdf5', 'r')

size_ratio=f.get('Size Ratio').value

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_size/sr_0_0.25/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_size/sr_0.25_0.5/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_size/sr_0.5_0.75/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_size/sr_0.75_1.0/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_size/sr_1.0_1.25/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_size/sr_1.25_1.5/')

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
