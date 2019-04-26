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
        