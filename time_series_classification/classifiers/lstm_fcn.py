# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
import tensorflow as tf
import numpy as np
import time 

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.layer_utils import AttentionLSTM

class Classifier_LSTMFCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
		self.output_directory = output_directory
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory+'model_init.hdf5')
		return

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		lstm_layer = Masking()(input_layer)
		lstm_layer = LSTM(64)(lstm_layer)
		lstm_layer = Dropout(0.8)(lstm_layer)

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

		x = concatenate([lstm_layer, gap_layer])

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(x)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def build_model(self, input_shape, nb_classes):
		ip = Input(input_shape)

		x = Masking()(ip)
		x = LSTM(64)(x)
		x = Dropout(0.8)(x)

		y = Permute((2, 1))(ip)
		y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
		y = BatchNormalization()(y)
		y = Activation('relu')(y)
		y = squeeze_excite_block(y)

		y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
		y = BatchNormalization()(y)
		y = Activation('relu')(y)
		y = squeeze_excite_block(y)

		y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
		y = BatchNormalization()(y)
		y = Activation('relu')(y)

		y = GlobalAveragePooling1D()(y)

		x = concatenate([x, y])

		out = Dense(nb_classes, activation='softmax')(x)

		model = Model(ip, out)
		model.summary()

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model
		
	def squeeze_excite_block(input):
		''' Create a squeeze-excite block
		Args:
			input: input tensor
			filters: number of output filters
			k: width factor
		Returns: a keras tensor
		'''
		filters = input._keras_shape[-1] # channel_axis = -1 for TF

		se = GlobalAveragePooling1D()(input)
		se = Reshape((1, filters))(se)
		se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
		se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
		se = multiply([input, se])
		return se


	def fit(self, x_train, y_train, x_val, y_val,y_true):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 8
		nb_epochs = 100

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = self.output_directory + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred