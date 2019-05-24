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
    
    
    def __init__(self, feature_values, DCfolder,TFRecord,feature,nclasses,N,name=""):
        self.name = name
        self.feature_values=feature_values
        self.DCfolder=DCfolder
        self.TFRecord=TFRecord
        self.feature=feature
        self.N=N
        self.nclasses=nclasses
        
        
    
    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def parameter_idx(self,class_value):
        value=str(class_value)
        feat_val=self.feature_values.astype(str)
        feat_val=list(feat_val)
        return feat_val.index(value)
    
    def image_process(self):

        picture_name_tensor = tf.placeholder(tf.string)
        picture_contents = tf.read_file(picture_name_tensor)
        picture = tf.image.decode_jpeg(picture_contents)
      
        return picture_name_tensor, picture

            
    def conversion(self): 
    
        #for iclas, clas in enumerate(os.listdir(self.DCfolder+self.feature)):
        for iclas, clas in enumerate(os.listdir(self.DCfolder+self.feature+'_'+str(self.nclasses)+'_'+str(self.N))):
            if clas=='.DS_Store':continue
            #one_clas= self.DCfolder+self.feature+'/'+clas+'/'
            one_clas= self.DCfolder+self.feature+'_'+str(self.nclasses)+'_'+str(self.N)+'/'+clas+'/'
            print('class',clas)
            class_value=float(clas.split('_')[-1])
                      
            pics = glob.glob(os.path.join(one_clas,"*.jpg"))
       
            #choosing files for Train Validation and Test 
            
            if len(pics)==0: continue #incase the class is empty
            
            valid_fraction = 0.15
            test_fraction = 0.15
            valid_number = int(len(pics)*valid_fraction)
            test_number = int(len(pics)*test_fraction)
            valid_pics = np.random.choice(np.array(pics), valid_number, replace=False)
            not_valid_pics = np.setdiff1d(pics,valid_pics)
            test_pics = np.random.choice(not_valid_pics, test_number, replace=False)
            train_pics = np.setdiff1d(not_valid_pics,test_pics)
            train_pics, valid_pics, test_pics = train_pics.tolist(), valid_pics.tolist(), test_pics.tolist()
            
#            os.mkdir(self.TFRecord+'Train/')
#            os.mkdir(self.TFRecord+'Test/')
#            os.mkdir(self.TFRecord+'Validation/')
##############################################################################################################################

            folder_name_train = self.TFRecord +'Train/'+str(class_value)+'.tfrecord'


            with tf.python_io.TFRecordWriter(folder_name_train) as Writer:
                for i_pic, pic_path in enumerate(train_pics):            
                    picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                    pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic_path} )                
                    
                    Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()),
                'idx': self._int64_feature(self.parameter_idx(class_value)) }))
               
                       
                    Writer.write(Example.SerializeToString())
                    
##############################################################################################################################              
            folder_name_valid= self.TFRecord + 'Validation/'+str(class_value)+'.tfrecord'

            with tf.python_io.TFRecordWriter(folder_name_valid) as Writer:
                for i_pic, pic_path in enumerate(valid_pics):            
                    picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                    pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic_path} )
                    
          
                    
                    Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()),
                'idx': self._int64_feature(self.parameter_idx(class_value)) }))                                                                                    
                    Writer.write(Example.SerializeToString())

##############################################################################################################################             
            folder_name_test = self.TFRecord+ 'Test/'+str(class_value)+'.tfrecord'
  
            with tf.python_io.TFRecordWriter(folder_name_test) as Writer:
                for i_pic, pic_path in enumerate(test_pics):            
                    picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                    pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic_path} )
                    
                    
                    Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()),
                'idx': self._int64_feature(self.parameter_idx(class_value)) }))                                                                             
                    Writer.write(Example.SerializeToString())
            
if __name__ == '__main__':

    converting_to_TFRecords= converting_to_TFRecords()    
    converting_to_TFRecords.conversion()      
            
        
 

 
    

