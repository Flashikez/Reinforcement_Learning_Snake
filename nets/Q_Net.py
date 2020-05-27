import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from nets.Net_Base import Net_Base

class Q_Net(Net_Base):
	def __init__(self, action_space, state_size, optimizer=keras.optimizers.Adam(lr=0.001)):
		super().__init__(action_space, state_size, optimizer)

	def shallow_copy(self):
		return Q_Net(self.action_space,self.state_size)


	def make_model(self,state_size,action_space,optimizer):
		model = tf.keras.Sequential()

		model.add(
			keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
								input_shape=state_size))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.MaxPooling2D(pool_size=2))
		model.add(keras.layers.Dropout(0.3))

		model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.MaxPooling2D(pool_size=2))
		model.add(keras.layers.Dropout(0.3))

		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(64, activation='relu'))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Dense(32, activation='relu'))
		model.add(keras.layers.Dropout(0.1))
		model.add(keras.layers.Dense(action_space.n, activation='softmax'))
		model.compile(optimizer=optimizer, loss='mse')

		return model





