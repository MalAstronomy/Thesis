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
from plot_confusion_matrix.py import ConfusionMatrix 


#from keras.initializers import glorot_uniform
from converting_images_to_TFRecords import converting_to_TFRecords as convertingTF
from extracting_images_from_TFRecords import extracting_TFRecords as extract



nclasses, nepochs, batch_size = 5, 30, 3           #
train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test= extract().handling_dataset(nepochs, batch_size) #, predict_iterator, steps_predict
input_shape= (224,224,3)


start=time.time()

class networks:
    
    def __init__(self, name=""):
        self.name = name
        
    def top_2_categorical_accuracy(self,y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=2) 
    
    def compiling(self, model, model_name):
        
        model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=[self.top_2_categorical_accuracy]) #kCA- K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))         
        return model
         

    def fitting(self, model, model_name): 
      
        #name= "paraview_"+model_name+"_"+str(nepochs)+"-{}".format(int(time.time()))
        name= "size_"+model_name+"_"+str(nepochs)+"-{}".format(int(time.time()))
        #name= "dchp_"+model_name+"_"+str(nepochs)+"-{}".format(int(time.time()))
        #callbacks=[TensorBoard(log_dir="/Users/malavikavijayendravasist/Desktop/mt2/Tensorboard/data_classes_handpicked/{}".format(name), batch_size=batch_size),
         #          ModelCheckpoint('/Users/malavikavijayendravasist/Desktop/mt2/Checkpoints/data_classes_handpicked/{}'.format(name)+'_{epoch:02d}.h5',monitor='val_acc',verbose=1,period=1)]  #-{val_accuracy:.2f}       
        callbacks=[TensorBoard(log_dir="/Users/malavikavijayendravasist/Desktop/mt2/Tensorboard/handpicked_size/{}".format(name), batch_size=batch_size),
                   ModelCheckpoint('/Users/malavikavijayendravasist/Desktop/mt2/Checkpoints/handpicked_size/{}'.format(name)+'_{epoch:02d}.h5',monitor='val_acc',verbose=1,period=1)]  #-{val_accuracy:.2f}       
        
        model.fit_generator(generator=train_iterator,
                            validation_data=valid_iterator,
                            steps_per_epoch=steps_per_epoch_train,
                            validation_steps=steps_per_epoch_valid,
                            epochs=nepochs,
                            callbacks=callbacks,
                            verbose=1,
                            workers=0)
        
        #confusion_matrix(valid_iterator.class_indices)
        model.save('/Users/malavikavijayendravasist/Desktop/mt2/Models/handpicked_size/{}'.format(name)+ '.hdf5')
        #model.save('/Users/malavikavijayendravasist/Desktop/mt2/Models/data_classes_handpicked/{}'.format(name)+ '.h5')
        #keras.models.load_model(/Users/malavikavijayendravasist/Desktop/mt2/Models/{}'.format(name)+ '.hdf5')
        
    def checks(self,model): 
        
        train_results= model.evaluate_generator(generator=train_iterator, steps=steps_per_epoch_train, max_queue_size=10, workers=0, use_multiprocessing=False)
        valid_results= model.evaluate_generator(generator=valid_iterator, steps=steps_per_epoch_valid, max_queue_size=10, workers=0, use_multiprocessing=False)
        test_results =model.evaluate_generator(generator=test_iterator, steps=steps_test, max_queue_size=10, workers=0, use_multiprocessing=False)
    
        print('Train loss:', train_results[0])
        print('Train accuracy:', train_results[1])
        print('Valid loss:', valid_results[0])
        print('Valid accuracy:', valid_results[1])
        print('Test loss:', test_results[0])
        print('Test accuracy:', test_results[1])
        
    
        #n_batches= 15
        #confusion_matrix( np.concatenate([np.argmax(valid_iterator[i][1], axis=1) for i in range(n_batches)]), np.argmax(model.predict_generator(valid_iterator, steps=n_batches), axis=1))
    def predict(self, model, model_name, pic_path): 
        
        param_names = ['mass_ratio']#,'size_ratio']
        param_values =[[0.25,0.5,0.75,1.0]]
        sess = tensorflow.Session()
        picture_file='/Users/malavikavijayendravasist/Desktop/mt2/prediction_images.txt'

        picture_names, size_ratios, mass_ratios = [], [], []
        with open(picture_file, 'rt') as f:
            for line in f:
                tmp1, tmp2 = line.split(",")
                picture_names.append(tmp1)
                mass_ratios.append(float(tmp2))
        
        picture_array = np.zeros((len(picture_names), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
        picture_name_tensor, picture= convertingTF().image_process(pic_path)

        for i,name in enumerate(picture_names):
            picture_array[i] = sess.run(picture, feed_dict={picture_name_tensor: name})
            picture_array[i] = np.array(picture_array[i], dtype=np.float32)
            picture_array[i] /= 255
            #if i%500==0: print(i)

        print("Start")
        predictions= model.predict(picture_array, verbose=1)
        print(predictions)
        print("End")
        
        perfect_counter = 0
        for i in range(len(picture_names)):
            print(i)
            max1 = np.amax(predictions[i][:4])
            idx1 = np.where(predictions[i]==max1)[0][0]
            pred1 = param_values[param_names.index('mass_ratio')][idx1] 
            print("Picture name:", picture_names[i])
            print("All values:", predictions[i])
            print("---|Predictions|--- Mass ratio: %.2f" % (pred1))
            if mass_ratios[i] == pred1:
                perfect_counter += 1
          
        print("Perfect accuracy:", float(perfect_counter)/float(len(picture_names)))
   
        #graph = tf.get_default_graph()
        #with graph.as_default():
        #    labels = model.predict(data)
        #model._make_predict_function(predict_iterator)#, steps= steps_predict)
        #model.predict(predict_iterator, verbose=1, steps= steps_predict)#verbose=1, callbacks=callbacks)
        
        #x = input_fn('data_rotated_resized_28_0.tfrecord.tfrecord')
        #with tensorflow.Session() as sess:  
        #    x_out = np.asarray(sess.run(x))
        #pred = model.predict(x_out,batch_size= batch_size, verbose=1)
        #print(pred)
        
        
    def printing(self):
        print('stuff')
        
    def saved_model(self, checkpoint_path): 
        

        #with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        #   model = load_model(checkpoint_path)
        model= load_model(checkpoint_path,custom_objects={'top_2_categorical_accuracy': self.top_2_categorical_accuracy})
        return model

    def fitting_resnet50(self, model_name):

         def resnet50():   
            model_input = layers.Input(shape=input_shape)
            model = ResNet50(input_tensor=model_input, include_top=True,weights='imagenet')
            last_layer = model.get_layer('avg_pool').output
            x= layers.Flatten(name='flatten')(last_layer)
            out = layers.Dense(nclasses, activation='softmax', name='output_layer')(x)
            custom_resnet_model = Model(inputs=model_input,outputs= out)
            #custom_resnet_model.summary()
    
            for layer in custom_resnet_model.layers[:-1]:
                layer.trainable = False
    
            custom_resnet_model.layers[-1].trainable
            return custom_resnet_model
            
         #resnet50().summary()
         return self.compiling(resnet50(),model_name)
        
        
    def fitting_mnist(self, model_name):
        
        def mnist(inputs, shape, nclass): 
       
            """
            Creates and returns neural net model
            """
            x = layers.Conv2D(32, kernel_size=(3, 3), 
                                  activation='relu',
                                  padding='valid',
                                  data_format=backend.image_data_format(),
                                  input_shape=shape)(inputs)
            x = layers.Dropout(0.5)(x)
            x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                  padding='valid')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), 
                                    strides=(2,2))(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            #x = layers.Dense(nclass, activation='sigmoid')(x)
            x = layers.Dense(nclass, activation='softmax')(x)
            return x
        
        model_input = layers.Input(shape=input_shape)
        model_output = mnist(model_input, input_shape, nclasses)
        model = Model(inputs=model_input, outputs=model_output)
        
        return self.compiling(model,model_name)
        
    
    
if __name__ == '__main__':
    #printing()
    network= networks()
    
    #model_name= 'mnist'
    #untrained_model= network.fitting_mnist(model_name)  #returns a compiled but untrained model 
    
    model_name= 'resnet50'
    untrained_model= network.fitting_resnet50(model_name)
    
    trained_model= network.fitting(untrained_model, model_name)
    
    #trained_model= network.saved_model(checkpoint_path= '/Users/malavikavijayendravasist/Desktop/mt2/Checkpoints/data_classes_handpicked/paraview_resnet50_10-1554367717_10.h5')
    #trained_model= network.fitting(trained_model, model_name)
    
    #print('im running')
    #network.checks(trained_model)
    #network.predict(trained_model,model_name, pic_path= '')
    end= time.time()
    print('time taken= ',end-start)
    
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
