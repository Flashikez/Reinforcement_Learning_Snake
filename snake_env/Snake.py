from collections import deque


class Snake():
	ACTION_LEFT = 0
	ACTION_RIGHT = 1
	ACTION_UP = 2
	ACTION_DOWN = 3
	def __init__(self,start_coordinates,initial_action):
		self.direction = initial_action
		self.head_coords = start_coordinates
		self.body = deque()



	def head_after_step(self,direction):
		'''

		'''
		new_head_coords = self.head_coords.copy()

		if direction == self.ACTION_LEFT:
			new_head_coords[1] -= 1
		elif direction == self.ACTION_RIGHT:
			new_head_coords[1] += 1
		elif direction == self.ACTION_UP:
			new_head_coords[0] -= 1
		else:
			new_head_coords[0] += 1
		# print(new_head_coords)
		return new_head_coords

	def step_non_food(self,coords):
		self.head_coords = coords

		if len(self.body) > 0:
			left = self.body.popleft()
			# self.body.append(left)
			return left
		else:
			return None
	def put_end(self,coords):
		self.body.append(coords)

	def step_food(self,food_coords):
		self.body.append(self.head_coords)
		self.head_coords = food_coords
		return self.head_coords


