#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 00:02:07 2019

@author: malavikavijayendravasist
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import tensorflow as tf
import numpy as np
import glob
import os


sess = tf.Session()


class converting_to_TFRecords: 
    
    def __init__(self, pic_path='/Users/malavikavijayendravasist/Desktop/mt2/images_resized',feature= 'Size Ratio',feat,self.TFRecord, name=""):
        self.name = name
        self.pic_path=pic_path
        self.feature=feature
        self.feat=feat
        self.TFRecord=TFRecord
        
    
    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _float64_feature(feature):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[feature]))
    
    def parameter_idx(self,param, value):
        idx = self.feature.index(param)
        return self.feature_values[idx].index(value)
    
    def image_process(self):

        picture_name_tensor = tf.placeholder(tf.string)
        picture_contents = tf.read_file(picture_name_tensor)
        picture = tf.image.decode_jpeg(picture_contents)
      
        return picture_name_tensor, picture
            
    def conversion(self): 
    
        f= self.feat 
        pics = glob.glob(os.path.join(self.pic_path,"*.jpg"))
   
        #choosing files for Train Validation and Test 

        valid_fraction = 0.15
        test_fraction = 0.15
        valid_number = int(len(pics)*valid_fraction)
        test_number = int(len(pics)*test_fraction)
        
        idx1 = np.random.choice(np.arange(len(pics)), valid_number, replace=False)
        valid_pics = np.array(pics)[idx1]
        valid_feat= f[idx1]
        not_valid_pics = np.setdiff1d(pics,valid_pics)
        not_valid_feat= np.setdiff1d(f,valid_feat)
        idx2=np.random.choice(np.arange(len(not_valid_pics)), test_number, replace=False)
        test_pics = not_valid_pics[idx2]
        test_feat=not_valid_feat[idx2]
        train_pics = np.setdiff1d(not_valid_pics,test_pics)
        train_feat= np.setdiff1d(not_valid_feat,test_feat)
        train_pics, valid_pics, test_pics = train_pics.tolist(), valid_pics.tolist(), test_pics.tolist()
        train_feat, valid_feat, test_feat = train_feat.tolist(), valid_feat.tolist(), test_feat.tolist()
    
        folder_name_train = self.TFRecord +'/Train/Train.tfrecord'


        with tf.python_io.TFRecordWriter(folder_name_train) as Writer:
            for i_pic, (pic,f) in enumerate(zip(train_pics,train_feat)):            
                picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic} )

                Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()), 'ratio': self._float64_feature(f)}))
                Writer.write(Example.SerializeToString())
        
        folder_name_valid= self.TFRecord + '/Validation/Validation.tfrecord'

        with tf.python_io.TFRecordWriter(folder_name_valid) as Writer:
            for i_pic, (pic,f) in enumerate(zip(valid_pics,valid_feat)):            
                picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic} )

                Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()), 'ratio': self._float64_feature(f)}))
                Writer.write(Example.SerializeToString())

        folder_name_test = self.TFRecord+ '/Test/Test.tfrecord'

        
        
        with tf.python_io.TFRecordWriter(folder_name_test) as Writer:
            for i_pic, (pic,f) in enumerate(zip(test_pics,test_feat)):            
                picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic} )

                Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()), 'ratio': self._float64_feature(f)}))
                Writer.write(Example.SerializeToString())
        
if __name__ == '__main__':

    converting_to_TFRecords= converting_to_TFRecords()    
    converting_to_TFRecords.conversion()      
            
        
 

 
    

