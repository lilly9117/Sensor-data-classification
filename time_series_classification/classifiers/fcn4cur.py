# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 

from utils.utils4cur import save_logs
from utils.utils4cur import calculate_metrics

# 모델 합치기
from keras import layers
from keras.models import Sequential, Model
from keras import Input
from keras.layers.merge import concatenate

class Classifier_FCN:
  
  def __init__(self, output_directory, input_shape1, input_shape2,  input_shape3, nb_classes, verbose=True,build=True):
    self.output_directory = output_directory
    if build == True:
      self.model = self.build_cur_model(input_shape1,input_shape2, input_shape3, nb_classes)
      if(verbose==True):
        self.model.summary()
      self.verbose = verbose
      self.model.save_weights(self.output_directory+'model_init.h5')
    return
    
  def build_cur_model(self, input_shape1, input_shape2, input_shape3, nb_classes):
    # R
    input_layerR = keras.layers.Input(input_shape1)
    conv1R = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layerR)
    conv1R = keras.layers.BatchNormalization()(conv1R)
    conv1R = keras.layers.Activation(activation='relu')(conv1R)
    conv2R = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1R)
    conv2R = keras.layers.BatchNormalization()(conv2R)
    conv2R = keras.layers.Activation('relu')(conv2R)
    conv3R = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2R)
    conv3R = keras.layers.BatchNormalization()(conv3R)
    conv3R = keras.layers.Activation('relu')(conv3R)
    gap_layerR = keras.layers.GlobalAveragePooling1D()(conv3R)
    
    model1 = gap_layerR
    # output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
    # model1 = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    #T
    input_layerT = keras.layers.Input(input_shape2)
    conv1T = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layerT)
    conv1T = keras.layers.BatchNormalization()(conv1T)
    conv1T = keras.layers.Activation(activation='relu')(conv1T)
    conv2T = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1T)
    conv2T = keras.layers.BatchNormalization()(conv2T)
    conv2T = keras.layers.Activation('relu')(conv2T)
    conv3T = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2T)
    conv3T = keras.layers.BatchNormalization()(conv3T)
    conv3T = keras.layers.Activation('relu')(conv3T)
    gap_layerT = keras.layers.GlobalAveragePooling1D()(conv3T)
    model2 = gap_layerT
    # output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
    # model2 = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    # S
    input_layerS = keras.layers.Input(input_shape3)
    conv1S = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layerS)
    conv1S = keras.layers.BatchNormalization()(conv1S)
    conv1S = keras.layers.Activation(activation='relu')(conv1S)
    conv2S = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1S)
    conv2S = keras.layers.BatchNormalization()(conv2S)
    conv2S = keras.layers.Activation('relu')(conv2S)
    conv3S = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2S)
    conv3S = keras.layers.BatchNormalization()(conv3S)
    conv3S = keras.layers.Activation('relu')(conv3S)
    gap_layerS = keras.layers.GlobalAveragePooling1D()(conv3S)
    model3 = gap_layerS
		# output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
    # model3 = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model = layers.concatenate([model1, model2, model3])
    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(model)
    model = keras.models.Model(inputs=[input_layerR, input_layerT, input_layerS], outputs=output_layer)  # 모델 정의 끝

    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])
      
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.00001)
    file_path = self.output_directory+'best_model.h5'
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
    self.callbacks = [reduce_lr,model_checkpoint]
    
    return model





  # def model_fit(self, x_train, y_train, x_val, y_val,y_true):
  def model_fit(self, Rx_train,Tx_train,Sx_train, y_train, Rx_test, Tx_test, Sx_test, y_test, y_true):
    if not tf.test.is_gpu_available:
      print('error')
      exit()
    # x_val and y_val are only used to monitor the test loss and NOT for training  
    batch_size = 64
    nb_epochs = 200
    mini_batch_size = int(min(Rx_train.shape[0]/10, batch_size))
    start_time = time.time() 

		# hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
		# 	verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

    hist = self.model.fit( [Rx_train, Tx_train, Sx_train], y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=([Rx_test,Tx_test,Sx_test],y_test), callbacks=self.callbacks)
      
    duration = time.time() - start_time
    
    self.model.save(self.output_directory+'last_model.h5')
    
    model = keras.models.load_model(self.output_directory+'best_model.h5')
    
    # y_pred = model.predict(x_val)
    
    y_pred = model.predict([Rx_test,Tx_test,Sx_test])
    
    # convert the predicted from binary to integer 
    y_pred = np.argmax(y_pred , axis=1)
    
    save_logs(self.output_directory, hist, y_pred, y_true, duration)
    
    keras.backend.clear_session()
    
  def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
    model_path = self.output_directory + 'best_model.h5'
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_test)
    
    if return_df_metrics:
      # y_pred = np.argmax(y_pred, axis=1)
      df_metrics = calculate_metrics(y_true, y_pred, 0.0)
      return df_metrics
    else:
      return y_pred