from MemoryBuffer import  Memory_Buffer
from tensorflow import keras
import numpy as np
class DQN_Agent():
	def __init__(self,QNet,action_space,gamma=0.95,epsilon=1,epsilon_min=0.1,epsilon_decay=0.995,memory_size = 2000,memory_batch_size = 32,target_model_update_iters = 1500,learning_steps = 500):

		self.QNet = QNet
		self.action_space = action_space
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.memory = Memory_Buffer(memory_size,memory_batch_size = memory_batch_size)
		# self.target_model = QNet
		self.target_model = self.QNet.shallow_copy()
		self.target_model_update_iters = target_model_update_iters
		self.learn_steps = 0
		self.learning_Steps = learning_steps
		self.steps_counter=  0

	def memorize(self,state,action,reward,new_state,done):
		self.memory.append((state,action,reward,new_state,done))


	def update_target_model(self):
		self.target_model.model.set_weights(self.QNet.model.get_weights())


	def get_action(self,state):

		if np.random.uniform() < self.epsilon:
			return self.action_space.sample()

		state = state[None,...]
		return self.QNet.predict_action(state)




	def learn(self):
		# self.steps_counter += 1
		if not self.memory.enough_samples() :
			return
		# if self.steps_counter != self.learning_Steps:
		# 	return

		self.epsilon = self.epsilon * self.epsilon_decay
		self.epsilon = max(self.epsilon,self.epsilon_min)

		# self.steps_counter = 0
		self.learn_steps += 1

		memory_samples = self.memory.sample()
		# print(memory_samples)
		states, actions_taken, rewards, new_states, dones = [np.array(l) for l in (zip(*memory_samples))]
		# print(actions_taken.shape)
		q_states = self.QNet.predict_raw(states)
		q_next_states = self.target_model.predict_raw(new_states)


		# print(q_states.shape)
		# print(q_next_states.shape)
		# input()
		for i in range(0,len(states)):
			# print(i)
			q_states[i,actions_taken[i]] = rewards[i] + self.gamma*np.max(q_next_states[i]) * (1-dones[i])

		self.QNet.train_on_batch(states,q_states)


		if self.learn_steps == self.target_model_update_iters:
			self.learn_steps = 0
			self.update_target_model()

	def save(self,path):
		self.QNet.save_model(path)

		# q_states =
		#
		# for sample in memory_samples:








