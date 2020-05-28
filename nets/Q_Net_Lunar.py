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

		# model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(256,input_shape=state_size, activation='relu'))
		# model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Dense(128, activation='relu'))
		# model.add(keras.layers.Dropout(0.1))
		# model.add(keras.layers.Dense(128, activation='relu'))
		model.add(keras.layers.Dense(action_space.n, activation='softmax'))
		model.compile(optimizer=optimizer, loss='mse')

		return model





