#!/bin/sh

#  data_classes_trial.py
#  
#
#  Created by Malavika Vijayendra Vasist on 26/04/2019.
#  

import numpy as np
import h5py
import os
import shutil
from os import path

f= h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/mergers_identified/mergers_28.hdf5', 'r')

size_ratio=f.get('Size Ratio').value

a=[]
#for n in os.listdir('/Users/malavikavijayendravasist/Desktop/mt2/paraview/images'):
for n in os.listdir('/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images'):
    a.append(int(((n.split('_')[1]).strip('\''))))
a=np.unique(a)
