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
    
    def __init__(self,TFRecord,feature_values,nclasses,dims,batch_size,nepochs,name=""):
        self.name = name
        self.TFRecord=TFRecord
        self.feature_values=feature_values
        self.nclasses=nclasses
        self.dims=dims    
        self.nepochs=nepochs
        self.batch_size=batch_size
    
    
    def handling_dataset(self):

        train_files= self.TFRecord + 'Train/'
        valid_files= self.TFRecord + 'Validation/'
        test_files = self.TFRecord + 'Test/'
   
        def extract_tfrec(tfrecord):
        
            features = {'picture_raw': tf.FixedLenFeature((), tf.string), 
                        'idx': tf.FixedLenFeature((), tf.int64)} #tf.float64
                        #'mass_ratio_idx': tf.FixedLenFeature((), tf.int64)} # Extract features using the keys set during creation
        
            parsed_features = tf.parse_single_example(tfrecord, features) # Extract the data record
            
            picture = tf.decode_raw(parsed_features['picture_raw'], tf.uint8) # or tf.io.decode_image- converting bytes to jpeg
            #picture /= 255.
            picture = tf.reshape(picture, [self.dims[0], self.dims[1], self.dims[2]])
            picture = tf.cast(picture, tf.float32) # The type is now uint8 but we need it to be float.
            
            features_final=tf.stack([tf.one_hot(parsed_features['idx'], self.nclasses)])                        
            features_final = tf.reshape(features_final, [-1])

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
        steps_per_epoch_train = int((npics_train+self.batch_size-1)/self.batch_size)-1
        
        
        npics_valid = 0
        for filename in os.listdir(valid_files):
            filename=valid_files+filename
            for record in tf.python_io.tf_record_iterator(filename):
                npics_valid += 1
        steps_per_epoch_valid = int((npics_valid+self.batch_size-1)/self.batch_size)-1
        
        npics_test = 0
        for filename in os.listdir(test_files):
            filename=test_files+filename
            for record in tf.python_io.tf_record_iterator(filename):
                npics_test += 1
        steps_test = int((npics_test+self.batch_size-1)/self.batch_size)-1
        
    
        train_files= [train_files+f  for f in os.listdir(train_files)]
        valid_files= [valid_files+f  for f in os.listdir(valid_files)]
        test_files= [test_files+f  for f in os.listdir(test_files)]

        train_dataset = dataset(train_files) #1986
        train_dataset = train_dataset.shuffle(buffer_size=npics_train)
        train_dataset = train_dataset.repeat(2*self.nepochs)
        train_dataset = train_dataset.batch(self.batch_size)
        train_iterator = make_iterator(train_dataset)
        
        valid_dataset = dataset(valid_files)
        valid_dataset = valid_dataset.shuffle(buffer_size=npics_valid)
        valid_dataset = valid_dataset.repeat(2*self.nepochs)
        valid_dataset = valid_dataset.batch(self.batch_size)
        valid_iterator = make_iterator(valid_dataset)
           
        test_dataset = dataset(test_files)
        test_dataset = test_dataset.shuffle(buffer_size=npics_test)
        test_dataset = test_dataset.repeat(2*self.nepochs)
        test_dataset = test_dataset.batch(self.batch_size)
        test_iterator = make_iterator(test_dataset)    
        
                   
        return train_iterator, valid_iterator, test_iterator, steps_per_epoch_train, steps_per_epoch_valid, steps_test
    
if __name__ == '__main__':

    extracting_TFRecords= extracting_TFRecords()    
    extracting_TFRecords.handling_dataset(TFRecord,feature_values,nclasses,dims,batch_size,nepochs)