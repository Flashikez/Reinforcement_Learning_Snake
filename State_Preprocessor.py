import numpy as np

class State_Preprocessor():
	def __init__(self,state_space,preprocess_functions):
		self.state_space = state_space
		self.preprocessing_functions = preprocess_functions
		self.output_shape = self.preprocess(np.random.random_sample(size=self.state_space.shape)).shape


	def preprocess(self, state):

		for func in self.preprocessing_functions:
			# print(state.shape)
			state = func(state)
		return state






def regionX(sliceX,image):
	return image[sliceX,...]

def to_gray(rgb):
	gray =  np.dot(rgb[...,:], [0.2989, 0.5870, 0.1140])
	return gray[...,None]