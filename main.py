from functools import partial

import matplotlib.pyplot as plt
import gym
from State_Preprocessor import State_Preprocessor
import State_Preprocessor as proccessor_funcs
# from nets.Q_Net import Q_Net
from nets.Q_Net_Lunar import Q_Net
from agents.DQN_Agent import DQN_Agent



env = gym.make('LunarLander-v2')
print(env.observation_space.shape)
env.reset()
# input()


# preprocessing_functions = [proccessor_funcs.to_gray,
# 						partial(proccessor_funcs.regionX,slice(25,155))
# ]
preprocessing_functions = []
preprocessor = State_Preprocessor(env.observation_space,preprocessing_functions)

preprocessed_state_shape = preprocessor.output_shape
print(preprocessed_state_shape)
print(env.action_space.n)

agent = DQN_Agent(Q_Net(env.action_space,preprocessed_state_shape),env.action_space)

episodes = 10000

episodes_rewards = []
for episode in range(episodes):
	actions_done = 0
	total_reward = 0
	current_state = env.reset()
	current_state = preprocessor.preprocess(current_state)
	done = False
	while not done:
		action = agent.get_action(current_state)
		# env.render()
		new_state, reward, done,_ = env.step(action)
		actions_done += 1
		# print(done)
		new_state = preprocessor.preprocess(new_state)
		total_reward += reward
		# print(actions_done,total_reward,done)
		agent.memorize(current_state,action,reward,new_state,done)
		agent.learn()


		current_state = new_state


	episodes_rewards.append(total_reward)
	print(f"Episode {episode} ended \t Reward: {total_reward} \t Average reward {sum(episodes_rewards) / len(episodes_rewards)}")

	if episode % 1000 == 0:
		print("Saving model")
		agent.save(f"after_{episode}_episodes.h5")



	# input()
	# print(state.shape)
	# print(env.action_space.n)
	# print(env.observation_space)
	# print(processed.shape)
	# plt.imshow(state[...,0], cmap='gray', vmin=0, vmax=1)
	# plt.show()

env.close()


