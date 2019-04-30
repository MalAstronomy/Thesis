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
#from tfrecords import converting_to_TFRecords as Ctfrecords
#from extracting_images_from_TFRecords_ import extracting_TFRecords as extractTF
#from extracting_ratio_records import  extracting_TFRecords as Ctfextract
#from networkss import networks
#from plot_confusion_matrix import ConfusionMatrix


sess = tensorflow.Session()

#training in Titan- /home/vasist/code/

class All: 
    
    def __init__(self, feature=None,pic_path=None,feature_values=None,DCfolder=None,epochs=None,batch_size=None,nclasses=None,dims=None,TBfolder=None,name=""):
        self.name = name
        self.feature='Size Ratio' #'Mass Ratio',
        self.feature_values=[]
        self.pic_path='/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images/' #'/home/vasist/images/'
        self.DCfolder='/Users/malavikavijayendravasist/Desktop/mt2/data_classes/data_classes_handpicked_size/'#'/home/vasist/data_classes/'
        #self.TFRecord='/home/vasist/TFRecords/data_classes/'
        self.TFRecord='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/trial/'#'/home/vasist/TFRecords/ratio/'
        self.feat=[]    # array of features of all the images in the same order as the images
        self.epochs=5
        self.batch_size=5
        self.nclasses=10
        self.dims=[224,224,3]
        self.TBfolder='/Users/malavikavijayendravasist/Desktop/mt2/Tensorboard/trial/'#'/home/vasist/Tensorboard/data_classes/'
        self.CPfolder='/Users/malavikavijayendravasist/Desktop/mt2/Checkpoints/trial/'#'/home/vasist/Checkpoints/data_classes/'
        self.Modelfolder='/Users/malavikavijayendravasist/Desktop/mt2/Models/trial/'#'/home/vasist/Models/data_classes/'
        self.model_name='mnist' #resnet50/mnist
        
    def Feature(self):
        
        images = os.listdir(self.pic_path)
        images=np.asarray(images)
        indices= np.random.choice(np.arange(len(images)),20) #len(images)
        print(indices)
        
        redshift=np.ndarray([])
        merger=np.ndarray([])
        angle=np.ndarray([])
        picture_names=np.ndarray([])

#        redshift=[]
#        merger=[]
#        angle=[]
#        picture_names=[]

        for i in indices:
 
            np.append(redshift,int(images[i].split('_')[1]))
            np.append(merger,int(images[i].split('_')[2]))
            np.append(angle,int(images[i].split('_')[3].split('.')[0]))
            np.append(picture_names,images[i])

        return redshift,merger,angle,picture_names
        
    def making_data_classes(self): #feature='Mass Ratio' #1
        
        redshift,merger,picture_names= self.Feature()
            
        making_classes= data_classes(self.pic_path,redshift,self.feature,merger,picture_names,self.DCfolder,self.nclasses)
        high=making_classes.making_classes()
        return high
        
    def making_tfrecords(self):    #1
        
        high=self.making_data_classes() 
        self.feature_values= np.linspace(0,high,self.nclasses+1)[1:] 
        self.feature_values=[round(i,3) for i in self.feature_values]
        
        convertingTF(self.feature_values, self.DCfolder,self.TFRecord).conversion()   
        
    def extracting_tfrecords(self): #1
        train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test= extractTF(self.TFRecord,self.feature_values,self.nclasses,self.dims,self.batch_size,self.nepochs).handling_dataset()
        
        return train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test
    
    def networkss(self): 
        
        train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test =self.extracting_tfrecords()
        
        network= networks(self.nclasses,self.nepochs,self.batch_size,train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test, self.model_name, self.feature,self.TBfolder,self.CPfolder,self.Modelfolder, self.dims)

        #untrained_model= network.fitting_mnist()  #returns a compiled but untrained model 
        untrained_model= network.fitting_resnet50()

        trained_model= network.fitting(untrained_model, self.model_name)

        #trained_model= network.saved_model()
        #trained_model= network.fitting(trained_model)   #to resume fitting 

        
        return trained_model
    
        #network.checks(trained_model)
        #network.predict(trained_model,model_name, pic_path= '')
        
     
    def ratio_records(self): #2

        redshift,feature,merger,picture_names=  self.Feature() 
        self.feat= data_classes(self.pic_path,redshift,feature,merger,picture_names,self.DCfolder).feature()    
        Ctfrecords(self.pic_path,feature,self.TFRecord).conversion()
            
    def making_datasets(self): #2
        
        Train_files= self.TFRecord+'Train/'
        Validation_files= self.TFRecord+'Validation/'
        Test_files= self.TFRecord+'Test/'
        
        Ctfextract(Train_files,Validation_files,Test_files, self.epochs,self.batch_size,self.nclasses)
        
        
    def predict(self,feature): 
        
        #picking 100 random images 
        images = os.listdir(self.pic_path)
        indices= np.random.randint(0,len(images),100)
        
        redshift,feature,merger,picture_names=Feature(feature)
        cl=np.linspace(0,high,self.nclasses+1) #self.feature_values doesnt include 0
        cl=[round(i,3) for i in cl]
            
        merger_p=merger[indices]
        picture_names_p=picture_names[indices]
        
        
        ytrue=[]
        for i in np.arange(len(merger_p)):
            for ind,c in enumerate(cl[1:]):
                ind+=1
                if merger_p[i] >= c1[ind-1] and merger_p[i] <= cl[ind]:
                    np.append(ytrue,c1[ind])
                    

        picture_array = np.zeros((len(picture_names_p), self.dims[0], self.dims[1], self.dims[2]), dtype=np.float32)
        picture_name_tensor, picture= convertingTF().image_process()

        for i,name in enumerate(picture_names_p):
            picture_array[i] = sess.run(picture, feed_dict={picture_name_tensor: name})
            picture_array[i] = np.array(picture_array[i], dtype=np.float32)
            picture_array[i] /= 255
            #if i%500==0: print(i)

        print("Start")
        predictions= model.predict(picture_array, verbose=1)
        print(predictions)
        print("End")
        
        ypred=[]
        perfect_counter = 0
        for i in range(len(picture_names_p)):
            max1 = np.amax(predictions[i][:self.nclasses])
            idx = np.where(predictions[i]==max1)[0][0]
            np.append(ypred, cl[idx+1])
           
        return ytrue,ypred 
    
    
    
    def ConfusionMatrix(self): 
        
        ytrue,ypred=self.predict(feature) #array of true and predicted class values 
        
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(ytrue, ypred, classes=self.feature_values.astype(str), title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plot_confusion_matrix(ytrue, ypred, classes=self.feature_values.astype(str), normalize=True, title='Normalized confusion matrix')
        
        

        
    def transformer(): 
        #save transformer
        pass
    
    def ytrueT(): 
        pass

    def inverse_transform(): 
        pass


if __name__ == '__main__':
    
    All=All()
    All.making_tfrecords()

#    high=All.data_classes()
#    tfrecords(high)

        
    
    
    
    
    
    
    
    
    
    
    
    
