import random

import numpy as np


class GameGrid():
	HEAD_VALUE_COL = np.array([0, 0, 255], dtype=np.float16)
	BODY_VALUE_COL = np.array([25,0,0], dtype=np.float16)
	EMPTY_VALUE_COL = np.array([0,127,0], dtype=np.float16)
	FOOD_VALUE_COL = np.array([255,0,0], dtype=np.float16)


	HEAD_VALUE = 3
	BODY_VALUE = 1
	EMPTY_VALUE = 0
	FOOD_VALUE = 2

	map = {HEAD_VALUE:HEAD_VALUE_COL,BODY_VALUE:BODY_VALUE_COL,EMPTY_VALUE:EMPTY_VALUE_COL,FOOD_VALUE:FOOD_VALUE_COL}


	def __init__(self, grid_size, unit_scale):
		self.grid_size = grid_size

		self.scale_factor = unit_scale
		self.width = self.grid_size[0] * self.scale_factor
		self.height = self.grid_size[1] * self.scale_factor
		self.channels = 3
		self.image_grid = np.zeros((self.width,self.height,self.channels) ,dtype=np.float16)
		self.image_grid[:,:,:] = self.EMPTY_VALUE_COL
		self.grid = np.full(grid_size,self.EMPTY_VALUE,dtype=np.float16)

	def init_snake(self, snake):
		self.place_tile(snake.head_coords, self.HEAD_VALUE)

		for part in snake.body:
			self.place_tile(part, self.BODY_VALUE)

	def place_tile(self, coord, value):
		self.grid[coord[0], coord[1]] = value
		x_start, x_end, y_start, y_end = self.coord_scale(coord)
		self.image_grid[x_start: x_end, y_start: y_end] = self.map[value]


	def erase_tile(self, coord):
		self.place_tile(coord, self.EMPTY_VALUE)
		x_start, x_end, y_start, y_end = self.coord_scale(coord)
		self.image_grid[x_start: x_end, y_start: y_end] = self.EMPTY_VALUE_COL

	def place_rand_food(self):
		coords = random.choice(np.argwhere(np.array(self.grid) == self.EMPTY_VALUE))
		self.place_tile(coords, self.FOOD_VALUE)

	def value_of(self, coord):
		# print(self.grid[coord[0], coord[1]])
		return self.grid[coord[0], coord[1]]

	def gameover(self, coords):
		out_of_border = coords[0] < 0 or coords[0] >= self.grid_size[0] or coords[1] < 0 or coords[1] >= self.grid_size[
			1]
		if out_of_border:
			return True
		body_tile = (self.value_of(coords) == self.BODY_VALUE)
		if body_tile:
			return True
		return False

	def reset_grid(self):
		self.image_grid = np.zeros((self.width, self.height, self.channels), dtype=np.float16)
		self.image_grid[:, :, :] = self.EMPTY_VALUE_COL
		self.grid = np.full(self.grid_size,self.EMPTY_VALUE,dtype=np.float16)

	def coord_scale(self, coords):
		x_start = coords[0] * self.scale_factor
		x_end = x_start + self.scale_factor
		y_start = coords[1] * self.scale_factor
		y_end = y_start + self.scale_factor
		
		return x_start,x_end,y_start,y_end
		


