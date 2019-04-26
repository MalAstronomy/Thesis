# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import tensorflow as tf
import numpy as np
import glob
import os

from PIL import Image


#param_names = ['mass_ratio']#,'size_ratio']
param_names = ['size_ratio']#,'mass_ratio']
#param_values =[[0.25, 0.5, 0.75, 1., 1.25]] #[0.25,0.5,0.75,1.0]]#
param_values =[[0.25, 0.5, 0.75, 1.]]   

#DataName='/home/vasist/TFRecords/data_classes_rotated_resized/data_rotated_resized_28.tfrecord'
#Folder='/home/vasist/data_classes_rotated_resized'

#DataName='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_rotated_resized/data_rotated_resized_28.tfrecord'
#Folder='/Users/malavikavijayendravasist/Desktop/mt2/data_classes_rotated_resized'

#DataName='/Users/malavikavijayendravasist/Desktop/mt2/paraview/TFRecords/28.tfrecord'
#Folder='/Users/malavikavijayendravasist/Desktop/mt2/paraview/data_classes'

DataName='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked_sizeqt/28.tfrecord'
Folder='/Users/malavikavijayendravasist/Desktop/mt2/data_classes/data_classes_handpicked_sizeqt'

#DataName='/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/28_dchp.tfrecord'
#Folder='/Users/malavikavijayendravasist/Desktop/mt2/data_classes_handpicked'

parameters=[0]*len(param_names)
parameter=[]
sess = tf.Session()

#os.mkdir("/"+split[1]+"/"+split[2]+"/"+split[3]+"/"+split[4]+"/Train")
#os.mkdir("/"+split[1]+"/"+split[2]+"/"+split[3]+"/"+split[4]+"/Validation")
#os.mkdir("/"+split[1]+"/"+split[2]+"/"+split[3]+"/"+split[4]+"/Test")


class converting_to_TFRecords: 
    
    def __init__(self,name=""):
        self.name = name
        
    
    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def parameter_idx(self,param, value):
        idx = param_names.index(param)
        #print(idx)
        return param_values[idx].index(value)
    
    def image_process(self):#,pic_path):
        
    
        #image = Image.open(pic_path)
        #image = image.resize((224, 224))
        #image_raw = np.array(image).tostring()

        picture_name_tensor = tf.placeholder(tf.string)
        picture_contents = tf.read_file(picture_name_tensor)
        picture = tf.image.decode_jpeg(picture_contents)
        
        #picture = tf.image.convert_image_dtype(picture, dtype=tf.float32)
        #picture= tf.expand_dims(picture, 0)
        #picture= tf.image.resize_bilinear(picture, [224, 224], align_corners=False)
        
        #image = tf.image.decode_jpeg(features['train/image'], channels=3)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #image = tf.expand_dims(image, 0)
        #image = tf.image.resize_bilinear(image, [299, 299], align_corners=False)
        
        return picture_name_tensor, picture
        #return picture_name_tensor, picture
    
    def image_process_(self,pic_path):
        with tf.gfile.FastGFile(pic_path, 'rb') as f:  #Read image data in terms of bytes
            #print(pic_path)
            image_data = f.read()    
            sess = tf.Session()
            decode_jpeg_data= tf.placeholder(dtype=tf.string)
            decode_jpeg= tf.image.decode_jpeg(decode_jpeg_data, channels=3)
                
            # Decode the RGB JPEG.
            image= sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
            return image
            #pic_raw=image if this function is used
            
    def conversion(self): 
    
        for iclas, clas in enumerate(os.listdir(Folder)): 
            one_clas= Folder+'/'+clas 
            
            #params= open(one_clas+'/params.txt','r')
            #for c,line in enumerate(params):
            #    parameters[c]=float(line)
            parameter=float(clas.split('_')[-1])
                
            
                      
            pics = glob.glob(os.path.join(one_clas,"*.jpg"))
            pics=np.random.choice(pics,20)
            print(iclas)
        #    for ifile,file in enumerate(os.listdir(one_clas)):
        
            #choosing files for Train Validation and Test 
            split = DataName.split('/')
            #print(split)
            valid_fraction = 0.15
            test_fraction = 0.15
            valid_number = int(len(pics)*valid_fraction)
            test_number = int(len(pics)*test_fraction)
            valid_pics = np.random.choice(np.array(pics), valid_number, replace=False)
            not_valid_pics = np.setdiff1d(pics,valid_pics)
            test_pics = np.random.choice(not_valid_pics, test_number, replace=False)
            train_pics = np.setdiff1d(not_valid_pics,test_pics)
            train_pics, valid_pics, test_pics = train_pics.tolist(), valid_pics.tolist(), test_pics.tolist()
        
            
            #folder_name_train = "/"+split[1]+"/"+split[2]+"/"+split[3]+"/"+split[4]+"/Train/"+split[5].split('.')[0]
            #folder_name_train += "_"+ str(iclas)+".tfrecord"
            
            #folder_name_train = '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_rotated_resized/Train/data_rotated_resized_28'+'_'+str(iclas)+'.tfrecord'
            #folder_name_train = '/Users/malavikavijayendravasist/Desktop/mt2/paraview/TFRecords/Train/'+str(iclas)+'.tfrecord'
            #folder_name_train = '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked/Train/'+str(iclas)+'.tfrecord'
            folder_name_train = '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked_sizeqt/Train/'+str(parameter)+'.tfrecord'


            with tf.python_io.TFRecordWriter(folder_name_train) as Writer:
                for i_pic, pic_path in enumerate(train_pics):            
                    picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                    pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic_path} )
                    
                    #pic_raw= self.image_process_(pic_path)
                    
                    Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()),
                 #       'mass_ratio_idx': self._int64_feature(self.parameter_idx('mass_ratio',parameters[0])) }))
                       'size_ratio_idx': self._int64_feature(self.parameter_idx('size_ratio',parameter))    }))
                    Writer.write(Example.SerializeToString())
            
            #folder_name_valid = "/"+split[1]+"/"+split[2]+"/"+split[3]+"/"+split[4]+"/Validation/"+split[5].split('.')[0]
            #folder_name_valid += "_"+ str(iclas)+".tfrecord"
            #folder_name_valid = '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_rotated_resized/Validation/data_rotated_resized_28'+'_'+str(iclas)+'.tfrecord'
            #folder_name_valid= '/Users/malavikavijayendravasist/Desktop/mt2/paraview/TFRecords/Validation/'+str(iclas)+'.tfrecord'
            #folder_name_valid= '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked/Validation/'+str(iclas)+'.tfrecord'
            folder_name_valid= '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked_sizeqt/Validation/'+str(parameter)+'.tfrecord'

            with tf.python_io.TFRecordWriter(folder_name_valid) as Writer:
                for i_pic, pic_path in enumerate(valid_pics):            
                    picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                    pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic_path} )
                    
                    #pic_raw= self.image_process_(pic_path)
                    
                    Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()),
                   #     'mass_ratio_idx': self._int64_feature(self.parameter_idx('mass_ratio',parameters[0])) })) 
                        'size_ratio_idx': self._int64_feature(self.parameter_idx('size_ratio',parameter))    }))
                    Writer.write(Example.SerializeToString())
            
            #folder_name_test = "/"+split[1]+"/"+split[2]+"/"+split[3]+"/"+split[4]+"/Test/"+split[5].split('.')[0]
            #folder_name_test += "_"+ str(iclas)+".tfrecord"
            #folder_name_test = '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_rotated_resized/Test/data_rotated_resized_28'+'_'+str(iclas)+'.tfrecord'
            #folder_name_test = '/Users/malavikavijayendravasist/Desktop/mt2/paraview/TFRecords/Test/'+str(iclas)+'.tfrecord'
            #folder_name_test = '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked/Test/'+str(iclas)+'.tfrecord'
            folder_name_test = '/Users/malavikavijayendravasist/Desktop/mt2/TFRecords/data_classes_handpicked_sizeqt/Test/'+str(parameter)+'.tfrecord'

            
            
            with tf.python_io.TFRecordWriter(folder_name_test) as Writer:
                for i_pic, pic_path in enumerate(test_pics):            
                    picture_name_tensor, image_tensor = self.image_process()  #to display the image- Image.show(Image.fromarray(np.asarray(image)))
                    pic_raw = sess.run(image_tensor, feed_dict={picture_name_tensor: pic_path} )
                    
                    #pic_raw= self.image_process_(pic_path)
                    
                    Example = tf.train.Example(features=tf.train.Features(feature={ 'picture_raw': self._bytes_feature(pic_raw.tostring()),
                        #'_ratio_idx': self._int64_feature(self.parameter_idx('mass_ratio',parameters[0])) })) 
                        'size_ratio_idx': self._int64_feature(self.parameter_idx('size_ratio',parameter))    }))
                    Writer.write(Example.SerializeToString())
            
if __name__ == '__main__':

    converting_to_TFRecords= converting_to_TFRecords()    
    converting_to_TFRecords.conversion()      
            
        
 

 
    

