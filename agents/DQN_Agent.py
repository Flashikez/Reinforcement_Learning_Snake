from MemoryBuffer import  Memory_Buffer
from tensorflow import keras
import numpy as np
class DQN_Agent():
	def __init__(self,QNet,action_space,gamma=0.95,epsilon=1,epsilon_min=0.1,epsilon_decay=1e-5,memory_size = 50000,memory_batch_size = 32,target_model_update_iters = 1500,learning_steps = 500):

		self.QNet = QNet
		self.action_space = action_space
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.memory = Memory_Buffer(memory_size)
		# self.target_model = QNet
		self.target_model = self.QNet.shallow_copy()
		self.current_steps = 0
		self.batch_size = memory_batch_size
		self.target_model_update = target_model_update_iters

	def memorize(self,state,action,reward,new_state,done):
		self.memory.store((state,action,reward,new_state,done))


	def update_target_model(self):
		self.target_model.model.set_weights(self.QNet.model.get_weights())


	def get_action(self,state):

		if np.random.uniform() < self.epsilon:
			return self.action_space.sample()

		state = state[None,...]
		return self.QNet.predict_action(state)




	def learn(self):
		# self.steps_counter += 1
		if  self.memory.size() < self.batch_size :
			return
		# if self.steps_counter != self.learning_Steps:
		# 	return

		state, action, reward, next_state, done = self.memory.sample(self.batch_size)


		self.current_steps += 1

		# for state, action, reward, next_state, done in batch:
		# 	# print(state)
		# 	state = state[None,...]
		# 	next_state = next_state[None,...]
		# 	target = reward
		# 	if not done:
		# 		target_predict = self.target_model.predict_raw(next_state)
		# 		# print(target_predict.shape)
		# 		# input()
		# 		target = reward + self.gamma * np.max(target_predict[0])
		# 	final_target = self.QNet.predict_raw(state)
		# 	final_target[0][action] = target
		# 	# print(state.shape)
		# 	# print(final_target.shape)
		# 	self.QNet.train_on_batch(state, final_target)


		if self.current_steps == self.target_model_update:
			self.update_target_model()
			self.current_steps = 0

		if self.epsilon > self.epsilon_min:
			self.epsilon -= self.epsilon_decay

	def save(self,path):
		self.QNet.save_model(path)

		# q_states =
		#
		# for sample in memory_samples:








