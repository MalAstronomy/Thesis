#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:06:33 2019

@author: malavikavijayendravasist
"""

import numpy as np
import h5py 
import os
import shutil
from os import path


#f= h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/mergers_identified/mergers_28.hdf5', 'r')
#size_ratio=f.get('Size Ratio').value

fs= h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/mergers_identified_qtscaled/mergers_28.hdf5', 'r')
size_ratio_scaled= fs.get('Size Ratio').value

a=[]
    #for n in os.listdir('/Users/malavikavijayendravasist/Desktop/mt2/paraview/images'):
for n in os.listdir('/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images'):
    a.append(int(((n.split('_')[1]).strip('\''))))
a=np.unique(a)
#a.append(int(n.split('.')[0]))
####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_0_0.25/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_0.25_0.5/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_0.5_0.75/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_0.75_1.0/')

####################################################################################################################
os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_1.0_1.25/')

####################################################################################################################
#os.mkdir('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_1.25_1.5/')

####################################################################################################################




for i in range(len(size_ratio_scaled)):
    
    if i not in a: continue
    else:
        if size_ratio_scaled[i] > 0 and size_ratio_scaled[i] < 0.25:
            for angle in range(0,360,15):
            
                src = path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images/28_' + str(i) + '_'+ str(angle) + '.jpg')
                dst=  path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_0_0.25/')
                shutil.copy(src,dst)
        
        elif size_ratio_scaled[i] > 0.25 and size_ratio_scaled[i] < 0.5:
            for angle in range(0,360,15):
            
                src = path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images/28_' + str(i) + '_'+ str(angle) + '.jpg')
                dst=  path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_0.25_0.5/')
                shutil.copy(src,dst)
        
        elif size_ratio_scaled[i] > 0.5 and size_ratio_scaled[i] < 0.75:
            for angle in range(0,360,15):
                src = path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images/28_' + str(i) + '_'+ str(angle) + '.jpg')
                dst=  path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_0.5_0.75/')
                shutil.copy(src,dst)
        
        elif size_ratio_scaled[i] > 0.75 and size_ratio_scaled[i] < 1.0:
            for angle in range(0,360,15):
                src = path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images/28_' + str(i) + '_'+ str(angle) + '.jpg')
                dst=  path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_0.75_1.0/')
                shutil.copy(src,dst)

        elif size_ratio_scaled[i] > 1.0 and size_ratio_scaled[i] < 1.25:
            for angle in range(0,360,15):
                src = path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images/28_' + str(i) + '_'+ str(angle) + '.jpg')
                dst=  path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_1.0_1.25/')
                shutil.copy(src,dst)
        
        elif size_ratio_scaled[i] > 1.25 and size_ratio_scaled[i] < 1.5:
            for angle in range(0,360,15):
                src = path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images/28_' + str(i) + '_'+ str(angle) + '.jpg')
                dst=  path.realpath('/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked_sizeqt/sr_1.25_1.5/')
                shutil.copy(src,dst)

