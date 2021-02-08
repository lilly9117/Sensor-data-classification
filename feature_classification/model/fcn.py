import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 

from sklearn.metrics import precision_score, f1_score, accuracy_score

class Classifier_FCN:

  def __init__(self,X_train,X_test,y_train,y_test):

    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test

    input_shape = self.X_train.shape[1:]
    nb_classes = 5

    self.model = self.build_model(input_shape, nb_classes)
    self.model.summary()

    self.output_directory = 'fcn_result/'

    self.model.save_weights(self.output_directory+'model_init.pt')
    return

  def build_model(self, input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
      metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
      min_lr=0.0001)

    file_path = 'fcn_result/'+'best_model.hdf5'

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
      save_best_only=True)

    self.callbacks = [reduce_lr,model_checkpoint]

    return model 

  def model_fit(self):
    if not tf.test.is_gpu_available:
      print('error')
      exit()
    # x_val and y_val are only used to monitor the test loss and NOT for training  
    batch_size = 32
    nb_epochs = 500

    mini_batch_size = int(min(self.X_train.shape[0]/10, batch_size))

    start_time = time.time() 

    hist = self.model.fit(self.X_train, self.y_train, batch_size=mini_batch_size, epochs=nb_epochs,
      verbose=False, validation_data=(self.X_test,self.y_test), callbacks=self.callbacks)
    
    duration = time.time() - start_time

    self.model.save(self.output_directory+'last_model.hdf5')

    model = keras.models.load_model(self.output_directory+'best_model.hdf5')

    y_pred = model.predict(self.X_test)

    # convert the predicted from binary to integer 
    y_pred = np.argmax(y_pred , axis=1)

    print('Accuracy Score:')
    print(accuracy_score(self.y_test,y_pred))

    keras.backend.clear_session()