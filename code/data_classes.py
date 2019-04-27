#!/bin/sh

#  data_classes.py
#  
#
#  Created by Malavika Vijayendra Vasist on 24/04/2019.
#  divides the images into their respective data classes


import numpy as np
import h5py
import os
from os import path

class data_classes:
    
    def __init__(self,pic_path,redshift,feature,merger,picture_names,DCfolder,nclasses):
        self.pic_path=pic_path
        self.redshift=redshift
        self.feature=feature
        self.merger=merger
        self.picture_names=picture_names
        self.N=len(merger) #no of images
        self.DCfolder=DCfolder
        self.nclasses=nclasses
    
    def feature(self):
        #feat holds the ratio values of all the images in the images folder in the same order as their names.
        feat=[]
        for i in np.arange(self.N):
            f= h5py.File('home/vasist/mergers_identified/mergers_'+str(self.redshift[i])+'.hdf5', 'r')
            feat=np.append(feat,f.get(self.feature).value[merger[i]])
        return feat
            
    def making_folders(self):
        
        #low= np.sort(feat)[0]
        os.mkdir(self.DCfolder+ feature+'/')
        
        high= np.sort(self.feature)[-1]

        cl = np.linspace(0,high,self.nclasses+1)
        cl=[round(i,3) for i in cl]
        
        for i,c in enumerate(cl[1:]):
            i+=1
            os.mkdir(self.DCfolder+ feature+ '/'+ str(cl[i-1])+ '_' + str(cl[i]))

        return high,cl

    def making_classes(self):
        
        high,cl=self.making_folders()
        
        for i in np.range(self.N):
            for ind,c in enumerate(cl[1:]):
                ind+=1
                if self.feature[i] >= c1[ind-1] and self.feature[i] <= cl[ind]:
                    src = path.realpath(self.pic_path + self.picture_names[i])
                    dst=  path.realpath(self.DCfolder+ feature+ '/'+ str(cl[ind-1])+ '_' + str(cl[ind])+ '/')
                    shutil.copy(src,dst)
    
        return high

