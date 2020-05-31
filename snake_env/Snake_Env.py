import numpy as np
import gym
from snake_env.Snake import Snake
from snake_env.GameGrid import GameGrid
from matplotlib import pyplot as plt

class Snake_Env(gym.Env):

	def __init__(self,grid_size = (20,20),unit_scale = 5, state_as_image = True,food_reward = 10,gameover_reward = -1):
		# self.headpos = np.random.randint(0,high=self.width,size=(2),dtype=np.uint8)
		self.grid = GameGrid(grid_size,unit_scale = unit_scale)
		self.gameover_reward = gameover_reward
		self.food_reward = food_reward
		self.action_space = gym.spaces.Discrete(4)
		self.state_as_img = state_as_image
		self.initialize_Game()
		self.renderer = None

		if self.state_as_img:
			self.observation_shape = self.grid.image_grid.shape
		else:
			self.observation_shape = self.grid.grid.shape


	def action_correction(self,last_action,current_action):
		if last_action == self.snake.ACTION_LEFT and current_action == self.snake.ACTION_RIGHT:
			return last_action
		if last_action == self.snake.ACTION_RIGHT and current_action == self.snake.ACTION_LEFT:
			return last_action
		if last_action == self.snake.ACTION_UP and current_action == self.snake.ACTION_DOWN:
			return last_action
		if last_action == self.snake.ACTION_DOWN and current_action == self.snake.ACTION_UP:
			return last_action

		return current_action


	def step(self,action):
		assert  self.action_space.contains(action)
		reward = 0

		action = self.action_correction(self.last_action,action)
		done = False
		old_head_pos = self.snake.head_coords
		new_head_pos = self.snake.head_after_step(action)
		# print(action)
		# print(old_head_pos,new_head_pos)
		# print('Before action')
		# print(self.last_state)

		if self.grid.gameover(new_head_pos):
			# print("OVER")
			reward = self.gameover_reward
			done = True

		elif self.grid.value_of(new_head_pos) == self.grid.FOOD_VALUE:
			reward = self.food_reward
			self.snake.step_food(new_head_pos)
			self.grid.place_tile(new_head_pos,self.grid.HEAD_VALUE)
			self.grid.place_tile(old_head_pos, self.grid.BODY_VALUE)
			self.grid.place_rand_food()



		elif self.grid.value_of(new_head_pos) == self.grid.EMPTY_VALUE:
			last_body_coord = self.snake.step_non_food(new_head_pos)
			# print(last_body_coord)
			if last_body_coord is None:
				self.grid.place_tile(old_head_pos, self.grid.EMPTY_VALUE)

			else:
				self.grid.erase_tile(last_body_coord)
				self.grid.place_tile(old_head_pos, self.grid.BODY_VALUE)
				self.snake.put_end(old_head_pos)

			self.grid.place_tile(new_head_pos, self.grid.HEAD_VALUE)
			# self.grid.place_tile(old_head_pos, self.grid.BODY_VALUE)
		self.last_action = action
		# print('AFter action')
		if self.state_as_img:
			self.last_state = self.grid.image_grid.copy()
		else:
			self.last_state = self.grid.grid.copy()
		# print(self.last_state)
		# print(self.last_state)
		return self.last_state,reward,done,None

	def reset(self):
		self.grid.reset_grid()
		self.initialize_Game()
		return self.last_state

	def initialize_Game(self):

		self.last_action = self.action_space.sample()

		self.snake = Snake(np.random.randint(0, high=self.grid.grid_size[0], size=(2), dtype=np.uint8), self.last_action)
		self.grid.init_snake(self.snake)
		self.grid.place_rand_food()
		if self.state_as_img:
			self.last_state = self.grid.image_grid.copy()
		else:
			self.last_state = self.grid.grid.copy()



	def render(self, mode='human', close=False, frame_speed=.1):
		if self.renderer is None:
			self.fig = plt.figure()
			self.renderer = self.fig.add_subplot(111)
			plt.ion()
			self.fig.show()

		self.renderer.clear()
		self.renderer.imshow(self.last_state.astype(np.uint8))
		plt.pause(frame_speed)
		self.fig.canvas.draw()
