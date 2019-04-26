#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:16:42 2019

@author: malavikavijayendravasist
"""
import tensorflow
from tensorflow.python.keras import layers 
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import top_k_categorical_accuracy 
from tensorflow.keras.models import load_model

from keras.utils import CustomObjectScope
from sklearn.metrics import confusion_matrix

import time
import numpy as np
import os
import h5py


#from keras.initializers import glorot_uniform

from data_classes import data_classes
from converting_images_to_TFRecords_ import converting_to_TFRecords as convertingTF
from tfrecords import converting_to_TFRecords as Ctfrecords
from extracting_images_from_TFRecords_ import extracting_TFRecords as extractTF
from extracting_ratio_records import  extracting_TFRecords as Ctfextract
from networkss import networks
from plot_confusion_matrix import ConfusionMatrix
 

features=['Mass Ratio','Size Ratio']
pic_path='/home/vasist/images/'

sess = tensorflow.Session()

#training in Titan- /home/vasist/code/

class All: 
    
    def __init__(self, features,pic_path,self.feature_values,DCfolder,epochs,batch_size,nclasses,dims,TBfolder,name=""):
        self.name = name
        self.features=['Mass Ratio','Size Ratio']
        self.feature_values=[]
        self.pic_path='/home/vasist/images/'
        self.DCfolder='/home/vasist/data_classes/'
        #self.TFRecord='/home/vasist/TFRecords/data_classes/'
        self.TFRecord='/home/vasist/TFRecords/ratio/'
        self.feat=[]    # array of features of all the images in the same order as the images
        self.epochs=epochs
        self.batch_size=batch_size
        self.nclasses=nclasses
        self.dims=[224,224,3]
        self.TBfolder='/home/vasist/Tensorboard/data_classes/'
        self.CPfolder='/home/vasist/Checkpoints/data_classes/'
        self.Modelfolder='/home/vasist/Models/data_classes/'
        self.model_name=model_name #resnet50/mnist
        
    def Feature(self,feature):
        
        images = os.listdir(self.pic_path)
        indices= len(images)
        
        redshift=[]
        merger=[]
        angle=[]
        picture_names=[]
                
        for i in indices: 
            redshift.append(int(images[i].split('_')[1])) 
            merger.append(int(images[i].split('_')[2])) 
            angle.append(int(s.split('_')[3].split('.')[0]))
            picture_names.append(images[i])
            
        return redshift,feature,merger,picture_names
        
    def making_data_classes(self): #feature='Mass Ratio' #1
        
        redshift,feature,merger,picture_names= self.Feature()
            
        making_classes= data_classes(self.pic_path,redshift,feature,merger,picture_names,self.DCfolder,self.nclasses)
        high=making_classes.making_classes()
        return high
        
    def making_tfrecords(self,high):    #1
        
        high=self.making_data_classes() 
        self.feature_values= np.linspace(0,high,self.nclasses+1)[1:] 

        convertingTF(self.feature_values, self.DCfolder,self.TFRecord).conversion()   
        
    def extracting_tfrecords(self): #1
        train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test= extractTF(self.TFRecord,self.feature_values,self.nclasses,self.dims,self.batch_size,self.nepochs).handling_dataset()
        
        return train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test
    
    def networkss(feature): 
        
        train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test =self.extracting_tfrecords()
        
        network= networks(self.nclasses,self.nepochs,self.batch_size,train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test, self.model_name, feature,self.TBfolder,self.CPfolder,self.Modelfolder, self.dims)

        #untrained_model= network.fitting_mnist()  #returns a compiled but untrained model 
        untrained_model= network.fitting_resnet50()

        trained_model= network.fitting(untrained_model, self.model_name)

        #trained_model= network.saved_model()
        #trained_model= network.fitting(trained_model)   #to resume fitting 

        #network.checks(trained_model)
        #network.predict(trained_model,model_name, pic_path= '')
        
    
        
        
        
        
    def ratio_records(self): #2

        redshift,feature,merger,picture_names=  self.Feature() 
        self.feat= data_classes(self.pic_path,redshift,feature,merger,picture_names,self.DCfolder).feature()    
        Ctfrecords(self.pic_path,feature,self.TFRecord).conversion()
            
    def making_datasets(self): 
        
        Train_files= self.TFRecord+'Train/'
        Validation_files= self.TFRecord+'Validation/'
        Test_files= self.TFRecord+'Test/'
        
        Ctfextract(Train_files,Validation_files,Test_files, self.epochs,self.batch_size,self.nclasses)
        
        
    def prediction_sample_space(pic_path): 
        
        #pick 100 random images
    
        images = os.listdir(pic_path)
        indices= np.random.randint(0,len(images),100)
        indx=[]
        picture_names=[]
            
        for i in indices: 
            indx.append(int(images[i].split('_')[1])) 
            picture_names.append(images[i])
            
        return indx, picture_names
        
    #def mass_ratio(): 
        
    #def size_ratio(): 
        
        
    #def ytrue(): 
        
    def transformer(): 
        #save transformer
        return transformer
    
    def ytrueT(): 
        return ytrueT
    
     
    

        
    def train():
        
    def save_model(): 
        
    def predict(): 
        return ypredT
    
    def inverse_transform(): 
    
    def confusion_matrix(): 
        
    
    
if __name__ == '__main__':
    All=All()
    high=All.data_classes()
    tfrecords(high)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
