

import random

class Memory_Buffer():

	def __init__(self, max_size,memory_batch_size = 32):
		self.memory = [None] * max_size
		self.max_size = max_size
		self.batch_size = memory_batch_size
		self.index = 0
		self.size = 0

	def append(self, data):
		self.memory[self.index] = data
		self.size = min(self.size + 1, self.max_size)
		self.index = (self.index + 1) % self.max_size

	def enough_samples(self):
		return self.size >= self.batch_size

	def sample(self):
		indexes = random.sample(range(self.size), self.batch_size)
		return [self.memory[index] for index in indexes]