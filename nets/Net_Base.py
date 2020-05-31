import numpy as np

from tensorflow import keras


class Net_Base():
	def __init__(self,action_space,state_size,optimizer = keras.optimizers.Adam(lr=0.001) ):
		self.state_size = state_size
		self.action_space = action_space
		self.model = self.make_model(self.state_size, self.action_space,optimizer)


	def save_model(self,path):
		self.model.save(path)


	def load_model(self,path):
		self.model.load(path)

	def make_model(self, state_size, action_space,optimizer):
		raise NotImplementedError("Override necessary")

	def shallow_copy(self):
		raise NotImplementedError("Override necessary")

	def train_on_batch(self,real,expected):
		self.model.fit(real,expected,verbose=0)

	def set_weights(self, weights):
		self.model.set_weights(weights)

	def predict_action(self, input_batch):
		probs = self.model.predict(input_batch)
		return np.argmax(probs)

	def predict_raw(self,input_batch):
		return self.model.predict(input_batch)