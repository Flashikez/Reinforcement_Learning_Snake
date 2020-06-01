from agents.DQN_Balaz import AgentDQN
from snake_env.Snake_Env import Snake_Env
from MemoryBuffer import  ExperienceReplay

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import trainings




def make_model( state_size, action_space, optimizer):
	model = tf.keras.Sequential()

	model.add(
		keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', activation='relu',
							input_shape=state_size))
	model.add(keras.layers.MaxPooling2D(pool_size=2))

	model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=2))

	model.add(keras.layers.Flatten())
	# model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dense(action_space))

	model.compile(optimizer=optimizer, loss='mse')
	return model
# 
#
episodes = 20000
save_every_nth_episode = 800

game_grid_size = (10,10)
env = Snake_Env(grid_size=game_grid_size,unit_scale = 5, state_as_image = True,food_reward = 10,gameover_reward = -1)
state_shape = env.observation_shape
agent = AgentDQN(0.99, env.action_space.n, make_model(state_shape,env.action_space.n,keras.optimizers.Adam(lr=0.001)),
                     ExperienceReplay(50000, state_shape), update_steps = 1000,
                     batch_size = 32)

trainings.train_agent(env,agent,episodes,save_every_nth_episode,'trainings/exp_1/')

def make_model( state_size, action_space, optimizer):
	model = tf.keras.Sequential()

	model.add(
		keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', activation='relu',
							input_shape=state_size))
	model.add(keras.layers.MaxPooling2D(pool_size=2))

	model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=2))

	model.add(keras.layers.Flatten())
	# model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dense(action_space, activation='softmax'))

	model.compile(optimizer=optimizer, loss='mse')
	return model
#
#
episodes = 10000
save_every_nth_episode = 800
game_grid_size = (10,10)
env = Snake_Env(grid_size=game_grid_size,unit_scale = 1, state_as_image = True,food_reward = 10,gameover_reward = -1)
state_shape = env.observation_shape
agent = AgentDQN(0.99, env.action_space.n, make_model(state_shape,env.action_space.n,keras.optimizers.Adam(lr=0.001)),
                     ExperienceReplay(50000, state_shape), update_steps = 1000,
                     batch_size = 32)

trainings.train_agent(env,agent,episodes,save_every_nth_episode,'trainings/exp_2/')

def make_model( state_size, action_space, optimizer):
	model = tf.keras.Sequential()

	model.add(
		keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', activation='relu',
							input_shape=state_size))
	# model.add(keras.layers.MaxPooling2D(pool_size=2))

	model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
	# model.add(keras.layers.MaxPooling2D(pool_size=2))

	model.add(keras.layers.Flatten())
	# model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dense(action_space, activation='softmax'))

	model.compile(optimizer=optimizer, loss='mse')
	return model
#
#
episodes = 10000
save_every_nth_episode = 800
game_grid_size = (10,10)
env = Snake_Env(grid_size=game_grid_size,unit_scale = 5, state_as_image = True,food_reward = 10,gameover_reward = -1)
state_shape = env.observation_shape
agent = AgentDQN(0.99, env.action_space.n, make_model(state_shape,env.action_space.n,keras.optimizers.Adam(lr=0.001)),
                     ExperienceReplay(50000, state_shape), update_steps = 1000,
                     batch_size = 32)

trainings.train_agent(env,agent,episodes,save_every_nth_episode,'trainings/exp_3/')


def play_env(env):
	def map_keyboard_controls(param):
		if param == 4:
			return 0
		if param == 6:
			return 1
		if param == 8:
			return 2
		if param == 2:
			return 3
		else: return 6
	
	while(True):
		total_reward = 0
		current_state = env.reset()
		done = False
		while not done:
			env.render()
			action = map_keyboard_controls(int(input()))
			new_state, reward, done, _ = env.step(action)
			# total_reward += reward
			# agent.remember(current_state,action,reward,new_state,done)
			# agent.learn()
			# current_state = new_state

# list = [5,5]
# print(list[0:0])
#
# play_env(env)