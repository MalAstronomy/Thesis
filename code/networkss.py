#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:45:30 2019

@author: malavikavijayendravasist
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:37:18 2019

@author: malavikavijayendravasist
"""
import tensorflow
from tensorflow.python.keras import layers 
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend
from tensorflow.keras.models import load_model
from keras.utils import CustomObjectScope
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy

import time
import numpy as np
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import ConfusionMatrix


#from keras.initializers import glorot_uniform
# from converting_images_to_TFRecords import converting_to_TFRecords as convertingTF
# from extracting_images_from_TFRecords import extracting_TFRecords as extract



start=time.time()

class networks:
    
    def __init__(self, nclasses,nepochs,batch_size,train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test, model_name, feat,TBfolder,CPfolder,Modelfolder, dims, name=""):
        self.name = name
        self.nclasses=nclasses
        self.nepochs=nepochs
        self.batch_size=batch_size
        self.train_iterator=train_iterator
        self.valid_iterator=valid_iterator
        self.test_iterator=test_iterator
        self.steps_per_epoch_train=steps_per_epoch_train
        self.steps_per_epoch_valid=steps_per_epoch_valid
        self.steps_test=steps_test
        self.model_name=model_name
        self.feat=feat
        self.TBfolder=TBfolder
        self.CPfolder=CPfolder
        self.Modelfolder=Modelfolder
        self.dims=tuple(dims)
        self.name=''
     
    def top_2_categorical_accuracy(self,y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=2) 
    
    def compiling(self, model):
        
        model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
        #model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=[self.top_2_categorical_accuracy]) 
        #kCA- K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))         
        return model
         

    def fitting(self, model): 
        
        f=self.feat.split(' ')[0]
        
        self.name= f + '_' + self.model_name+ "_" + str(self.nepochs)+ "_" + str(int(time.time()))
        print(self.name)
        
        l=self.nepochs
        c=0
        while l>0:
            l//=10
            c+=1
        a="_{epoch:0"+str(c)+"d}.h5"
        
        callbacks=[TensorBoard(log_dir=self.TBfolder+self.name, batch_size=self.batch_size), ModelCheckpoint(self.CPfolder + self.name + a,monitor='val_acc',verbose=1,period=1)]  #-{val_accuracy:.2f}
        
        model.fit_generator(generator=self.train_iterator,
                            validation_data=self.valid_iterator,
                            steps_per_epoch=self.steps_per_epoch_train,
                            validation_steps=self.steps_per_epoch_valid,
                            epochs=self.nepochs,
                            callbacks=callbacks,
                            verbose=1,
                            workers=0)

        model.save(self.Modelfolder + self.name + '.hdf5')
        return self.name
        
    def checks(self,model): 
        
        train_results= model.evaluate_generator(generator=self.train_iterator, steps=self.steps_per_epoch_train, max_queue_size=10, workers=0, use_multiprocessing=False)
        valid_results= model.evaluate_generator(generator=self.valid_iterator, steps=self.steps_per_epoch_valid, max_queue_size=10, workers=0, use_multiprocessing=False)
        test_results =model.evaluate_generator(generator=self.test_iterator, steps=self.steps_test, max_queue_size=10, workers=0, use_multiprocessing=False)
    
        print('Train loss:', train_results[0])
        print('Train accuracy:', train_results[1])
        print('Valid loss:', valid_results[0])
        print('Valid accuracy:', valid_results[1])
        print('Test loss:', test_results[0])
        print('Test accuracy:', test_results[1])
               
    def fitting_mnist(self):
        
        def mnist(inputs): 
       
            """
            Creates and returns neural net model
            """
        
            x = layers.Conv2D(32, kernel_size=(3, 3), 
                                  activation='relu',
                                  padding='valid',
                                  data_format=backend.image_data_format(),
                                  input_shape=self.dims)(inputs)
            x = layers.Dropout(0.5)(x)
            x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',padding='valid')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            #x = layers.Dense(self.nclasses, activation='sigmoid')(x)
            x = layers.Dense(self.nclasses, activation='softmax')(x)
            return x
        
        model_input = layers.Input(shape=self.dims)
        model_output = mnist(model_input)
        model = Model(inputs=model_input, outputs=model_output)
        
        return self.compiling(model)
    

    def fitting_resnet50(self):

        def resnet50():
            model_input = layers.Input(shape=self.dims)
            model = ResNet50(input_tensor=model_input, include_top=True,weights='imagenet')
            last_layer = model.get_layer('avg_pool').output
            x= layers.Flatten(name='flatten')(last_layer)
            out = layers.Dense(self.nclasses, activation='softmax', name='output_layer')(x)
            custom_resnet_model = Model(inputs=model_input,outputs= out)
    
            for layer in custom_resnet_model.layers[:-1]:
                layer.trainable = False
    
            custom_resnet_model.layers[-1].trainable
            return custom_resnet_model
            
        return self.compiling(resnet50())
        
    def saved_model(self): 

        model= load_model(self.CPfolder+self.name+'_'+str(self.nepochs)+'.h5')
        #model= load_model(self.CPfolder,custom_objects={'top_2_categorical_accuracy': self.top_2_categorical_accuracy})
        return model    
    
if __name__ == '__main__':
    #printing()
    network= networks()
    
    #model_name= 'mnist'
    #untrained_model= network.fitting_mnist()  #returns a compiled but untrained model 
    
    #model_name= 'resnet50'
    untrained_model= network.fitting_resnet50()
    
    trained_model= network.fitting(untrained_model)
    
    #trained_model= network.saved_model()
    #trained_model= network.fitting(trained_model)
    
    #print('im running')
    #network.checks(trained_model)
    #network.predict(trained_model,model_name, pic_path= '')
    end= time.time()
    print('time taken= ',end-start)
    
    
    

        
        
        
        
        
