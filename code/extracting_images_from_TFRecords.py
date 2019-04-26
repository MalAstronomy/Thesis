#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:46:10 2019

@author: malavikavijayendravasist
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:32:30 2019

@author: malavikavijayendravasist
"""
import tensorflow as tf
import os
from tensorflow.python.keras import backend

#from keras import backend



class extracting_TFRecords:
    
    def __init__(self, name=""):
        self.name = name
    
    
    def handling_dataset(self, nepochs, batch_size):
        
        #train_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_rotated_resized/Train/'
        #valid_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_rotated_resized/Validation/'
        #test_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_rotated_resized/Test/'
        #predict_files= '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_rotated_resized/Predict/'

        #train_files='/Users/malavikavijayendravasist/Desktop/mt2/paraview/TFRecords/Train/'
        #valid_files='/Users/malavikavijayendravasist/Desktop/mt2/paraview/TFRecords/Validation/'
        #test_files='/Users/malavikavijayendravasist/Desktop/mt2/paraview/TFRecords/Test/'
        #predict_files= '/Users/malavikavijayendravasist/Desktop/mt2/paraview/TFRecords/data_classes_rotated_resized/Predict/'


        train_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked_sizeqt/Train/'
        valid_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked_sizeqt/Validation/'
        test_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked_sizeqt/Test/'
        
        #train_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked/Train/'
        #valid_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked/Validation/'
        #test_files='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked/Test/'


        #nclasses, nepochs, batch_size = 4, 10, 128
            
        param_names = ['size_ratio'] #'mass_ratio']#,
        param_values =[[0.25, 0.5, 0.75, 1.0,1.25]] #[[0.25,0.5,0.75,1.0]]#,
        
        dims=[224,224,3]
        
        def get_len(param):
            idx = param_names.index(param)
            return len(param_values[idx])
    
        def extract_tfrec(tfrecord):
        
            features = {'picture_raw': tf.FixedLenFeature((), tf.string), 
                        'size_ratio_idx': tf.FixedLenFeature((), tf.int64)} #tf.float64
                        #'mass_ratio_idx': tf.FixedLenFeature((), tf.int64)} # Extract features using the keys set during creation
        
            parsed_features = tf.parse_single_example(tfrecord, features) # Extract the data record
            
            picture = tf.decode_raw(parsed_features['picture_raw'], tf.uint8) # or tf.io.decode_image- converting bytes to jpeg
            #picture /= 255.
            picture = tf.reshape(picture, [dims[0], dims[1], dims[2]])
            picture = tf.cast(picture, tf.float32) # The type is now uint8 but we need it to be float.
            
            features_final=tf.stack([tf.one_hot(parsed_features['size_ratio_idx'], get_len('size_ratio'))]) #,
                                     #tf.one_hot(parsed_features['mass_ratio_idx'], get_len('mass_ratio'))])
                                          
            features_final = tf.reshape(features_final, [-1])
            
            #print(features_final)
            return picture, features_final
    
        
        def dataset(files):  # Pipeline of dataset and iterator 
            dataset = tf.data.TFRecordDataset(files) #train_files, valid_files, test_files
            return dataset.map(map_func=extract_tfrec) #per element transformation
    
        def make_iterator(dataset): #iterating over all the files in the dataset
            iterator = dataset.make_one_shot_iterator()
            next_val = iterator.get_next()
            with backend.get_session().as_default() as sess:
                
                while True:
                    try:
                        inputs, labels = sess.run(next_val)
                        yield inputs, labels
                    except tf.errors.OutOfRangeError:
                        break
                    
                    
                
    
        #no of pics in each train, valid and test         
        npics_train = 0
        for filename in os.listdir(train_files):
            filename=train_files+filename
            for record in tf.python_io.tf_record_iterator(filename):
                npics_train += 1
        steps_per_epoch_train = int((npics_train+batch_size-1)/batch_size)-1
        
        
        npics_valid = 0
        for filename in os.listdir(valid_files):
            filename=valid_files+filename
            for record in tf.python_io.tf_record_iterator(filename):
                npics_valid += 1
        steps_per_epoch_valid = int((npics_valid+batch_size-1)/batch_size)-1
        
        npics_test = 0
        for filename in os.listdir(test_files):
            filename=test_files+filename
            for record in tf.python_io.tf_record_iterator(filename):
                npics_test += 1
        steps_test = int((npics_test+batch_size-1)/batch_size)-1
        
        
        #npics_predict=0
        #or filename in os.listdir(predict_files):
        #    filename=predict_files+filename
        #    for record in tf.python_io.tf_record_iterator(filename):
        #        npics_predict += 1
        #steps_predict = int((npics_predict+batch_size-1)/batch_size)-1
            
    
        train_files= [train_files+f  for f in os.listdir(train_files)]
        valid_files= [valid_files+f  for f in os.listdir(valid_files)]
        test_files= [test_files+f  for f in os.listdir(test_files)]
        #predict_files= [predict_files+f  for f in os.listdir(predict_files)]
        
        train_dataset = dataset(train_files) #1986
        train_dataset = train_dataset.shuffle(buffer_size=npics_train)
        train_dataset = train_dataset.repeat(2*nepochs)
        train_dataset = train_dataset.batch(batch_size)
        train_iterator = make_iterator(train_dataset)
        
        valid_dataset = dataset(valid_files)
        valid_dataset = valid_dataset.shuffle(buffer_size=npics_valid)
        valid_dataset = valid_dataset.repeat(2*nepochs)
        valid_dataset = valid_dataset.batch(batch_size)
        valid_iterator = make_iterator(valid_dataset)
           
        test_dataset = dataset(test_files)
        test_dataset = test_dataset.shuffle(buffer_size=npics_test)
        test_dataset = test_dataset.repeat(2*nepochs)
        test_dataset = test_dataset.batch(batch_size)
        test_iterator = make_iterator(test_dataset)    
        
        #predict_dataset= dataset(predict_files) 
        #predict_dataset = predict_dataset.shuffle(buffer_size=npics_test)
        #predict_dataset = predict_dataset.repeat(2*nepochs)
        #predict_dataset = predict_dataset.batch(batch_size)
        #predict_iterator = make_iterator(predict_dataset) 
        
            
        return train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test#, predict_iterator, steps_predict
    
if __name__ == '__main__':

    train_record='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked_sizeqt/Train/0.tfrecord'
    
    extracting_TFRecords= extracting_TFRecords()    
    extracting_TFRecords.extract_tfrec(train_record)