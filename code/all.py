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
from extracting_images_from_TFRecords_ import extracting_TFRecords as extractTF
#from extracting_ratio_records import  extracting_TFRecords as Ctfextract
from networkss import networks
from plot_confusion_matrix import ConfusionMatrix


sess = tensorflow.Session()

#training in Titan- /home/vasist/code/

class All:
    
    def __init__(self, feature=None,pic_path=None,feature_values=None,DCfolder=None,epochs=None,batch_size=None,nclasses=None,dims=None,TBfolder=None,name=""):
        self.name = name
        self.feature='Size Ratio' #'Mass Ratio',
        self.feature_values=[]
        self.pic_path= '/home/vasist/images/images_resized/' #'/Users/malavikavijayendravasist/Desktop/mt2/handpicked_images/'
        self.DCfolder='/home/vasist/data_classes/' # /Users/malavikavijayendravasist/Desktop/mt2/data_classes/data_classes_trial/'
        #self.TFRecord='/home/vasist/TFRecords/data_classes/'
        self.TFRecord='/home/vasist/TFRecords/ratio/' #'/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/trial/'
        self.feat=[]    # array of features of all the images in the same order as the images
        self.nepochs=5
        self.batch_size=5
        self.nclasses=10
        self.dims=[224,224,3]
        self.TBfolder='/home/vasist/Tensorboard/data_classes/' #'/Users/malavikavijayendravasist/Desktop/mt2/Tensorboard/trial/'
        self.CPfolder='/home/vasist/Checkpoints/data_classes/' #'/Users/malavikavijayendravasist/Desktop/mt2/Checkpoints/trial/'
        self.Modelfolder= '/home/vasist/Models/data_classes/' #'/Users/malavikavijayendravasist/Desktop/mt2/Models/trial/'
        self.network_name='mnist' # resnet50/mnist
    
    
    def Feature(self):
        
        images = os.listdir(self.pic_path)
        images=np.asarray(images)
        
        indices= np.random.choice(np.arange(len(images)),100) #len(images)
        
        
        redshift=[]
        merger=[]
        angle=[]
        picture_names=[]
        
        for i in indices:
            redshift.append(int(images[i].split('_')[1]))
            merger.append(int(images[i].split('_')[2]))
            angle.append(int(images[i].split('_')[3].split('.')[0]))
            picture_names.append(images[i])
        
        return redshift,merger,picture_names
    
    def making_data_classes(self): #feature='Mass Ratio' #1
        
        redshift,merger,picture_names= self.Feature()
        
        making_classes= data_classes(self.pic_path,redshift,self.feature,merger,picture_names,self.DCfolder,self.nclasses)
        high=making_classes.making_classes()
        
        f=h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/high.hdf5','w')
        f.create_dataset('high',data=high)
        f.close()
    
    
    def making_tfrecords(self):    #1
        
        #self.high=self.making_data_classes()
        f=h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/high.hdf5','r')
        high=f['high'].value
        f.close()
        self.feature_values= np.linspace(0,high,self.nclasses+1)[1:]
        self.feature_values=np.asarray([round(i,3) for i in self.feature_values])
        convertingTF(self.feature_values, self.DCfolder,self.TFRecord,self.feature).conversion()
    
    
    
    def extracting_tfrecords(self): #1
        train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test= extractTF(self.TFRecord,self.feature_values,self.nclasses,self.dims,self.batch_size,self.nepochs).handling_dataset()
        
        return train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test
    
    
    def networkss(self):
        
        train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test =self.extracting_tfrecords()
        
        network= networks(self.nclasses,self.nepochs,self.batch_size,train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test, self.network_name, self.feature,self.TBfolder,self.CPfolder,self.Modelfolder, self.dims)
        
        #untrained_model= network.fitting_mnist()  #returns a compiled but untrained model
        untrained_model= network.fitting_resnet50()
        model_name=network.fitting(untrained_model) #the model is saved here
        
        f= h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/model_name.hdf5','w')
        f.create_dataset('model_name',data=model_name)
        f.close()
    
    #trained_model= network.fitting(trained_model)   #to resume fitting
    
    def saved_model(self):
        
        f= h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/model_name.hdf5','r')
        model_name= f['model_name'].value
        f.close()
        #         print(model_name)
        
        #         trained_model_name=self.Modelfolder+model_name
        #         print(trained_model_name)
        model= load_model(self.CPfolder+model_name+'_'+str(self.nepochs)+'.h5')
        #model= load_model(self.CPfolder,custom_objects={'top_2_categorical_accuracy': self.top_2_categorical_accuracy})
        return model
    
    
    def predict(self):
        
        #images = os.listdir(self.pic_path)
        
        ###picking 100 random images and selecting 10 out of them
        
        redshift,merger,picture_names=self.Feature() #100 random images
        making_classes= data_classes(self.pic_path,redshift,self.feature,merger,picture_names,self.DCfolder,self.nclasses)
        feat=making_classes.feat() #feature values for 100 random images obtained
        
        #########################selecting 10
        
        indices= np.random.randint(0,len(picture_names),10)
        merger=np.asarray(merger)
        picture_names=np.asarray(picture_names)
        merger_p=merger[indices]
        picture_names_p=picture_names[indices]
        feat_p=feat[indices]
        print('fp',feat_p)
        ######################################
        
        
        cl=self.cl()
        print(cl)
        ytrue=[]
        for i in np.arange(len(merger_p)):
            for ind,c in enumerate(cl[1:]):
                ind+=1
                if feat_p[i] >= cl[ind-1] and feat_p[i] <= cl[ind]:
                    #np.append(ytrue,cl[ind])
                    ytrue.append(cl[ind])
                    break
        
        
        
        picture_array = np.zeros((len(picture_names_p), self.dims[0], self.dims[1], self.dims[2]), dtype=np.float32)
        picture_name_tensor, picture= convertingTF(self.feature_values, self.DCfolder,self.TFRecord,self.feature).image_process()
        
        for i,name in enumerate(picture_names_p):
            Name=self.pic_path+name
            picture_array[i] = sess.run(picture, feed_dict={picture_name_tensor: Name})
            picture_array[i] = np.array(picture_array[i], dtype=np.float32)
            picture_array[i] /= 255
                    #if i%500==0: print(i)
                    
                    
        model= self.saved_model()
        print("Start")
        predictions= model.predict(picture_array, verbose=1)
        print(predictions)
        print("End")
        
        ypred=[]
        perfect_counter = 0
        for i in range(len(picture_names_p)):
            max1 = np.amax(predictions[i][:self.nclasses])
            idx = np.where(predictions[i]==max1)[0][0]
            ypred.append(cl[idx+1])
                    
                    
        print(merger_p)
        print(picture_names_p)
        print(ytrue)
        print(ypred)


        f=h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/CM.hdf5','w')
        s1=f.create_dataset('ytrue',data=ytrue)
        s2=f.create_dataset('ypred',data=ypred)
        f.close()
    
        return ytrue,ypred
    
    def cl(self):
        
        f=h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/high.hdf5','r')
        high=f['high'].value
        f.close()
        cl=np.linspace(0,high,10+1) #self.feature_values doesnt include 0
        cl=[round(i,3) for i in cl]
        return cl

    def get_index(self,ytrue, ypred):
        
        cl=self.cl()
        index_ytrue=np.zeros(len(ytrue))
        index_ypred=np.zeros(len(ypred))
        
        for i in np.arange(len(ytrue)):
            index_ytrue[i]=np.where(ytrue[i]==cl)[0]-1
            index_ypred[i]=np.where(ypred[i]==cl)[0]-1
        return index_ytrue,index_ypred


    def ConfusionMatrix(self):
        #array of true and predicted class values
        f=h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/CM.hdf5','r')
        ytrue=f['ytrue'].value
        ypred=f['ypred'].value
        f.close()
        
        cl=self.cl()[1:]
        cl=np.asarray(cl)
        index_ytrue,index_ypred= self.get_index(ytrue,ypred)
        print(index_ytrue,index_ypred)
        CM=ConfusionMatrix()
        #print('fv',self.feature_values)
        # Plot non-normalized confusion matrix
        print(cl.astype(str))
        CM.plot_confusion_matrix(index_ytrue, index_ypred, classes=cl.astype(str), title='Confusion matrix, without normalization')
        
        # Plot normalized confusion matrix
        CM.plot_confusion_matrix(index_ytrue, index_ypred, classes=cl.astype(str), normalize=True, title='Normalized confusion matrix')
      
    def transformer(): 
        #save transformer
        pass
    
    def ytrueT(): 
        pass

    def inverse_transform(): 
        pass


if __name__ == '__main__':
    
    All=All()
    a1=All.making_data_classes
    a2=All.making_tfrecords
    a3=All.extracting_tfrecords
    a4=All.networkss
    a5=All.predict
    a6=All.ConfusionMatrix


        
    
    
    
    
    
    
    
    
    
    
    
    
