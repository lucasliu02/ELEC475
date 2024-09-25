

import torch
import torch.nn.functional as F
import torch.nn as nn


# steps 4 & 5
class autoencoderMLP4Layer(nn.Module):

	def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
		super(autoencoderMLP4Layer, self).__init__()
		N2 = 392
		self.fc1 = nn.Linear(N_input, N2)		# input = 1x784, output = 1x392
		self.fc2 = nn.Linear(N2, N_bottleneck)	# output = 1xN
		self.fc3 = nn.Linear(N_bottleneck, N2)	# output = 1x392
		self.fc4 = nn.Linear(N2, N_output)		# output = 1x784
		self.type = 'NLP4'
		self.input_shape = (1, 28*28)

	def forward(self, X):
		# encoder
		X = self.fc1(X)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)

		# decoder
		X = self.fc3(X)
		X = F.relu(X)
		X = self.fc4(X)
		X = torch.sigmoid(X)

		return X

# step 6
class autoencoderMLP4Layer_Interpolator(nn.Module):

	def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
		super(autoencoderMLP4Layer_Interpolator, self).__init__()
		N2 = 392
		self.fc1 = nn.Linear(N_input, N2)		# input = 1x784, output = 1x392
		self.fc2 = nn.Linear(N2, N_bottleneck)	# output = 1xN
		self.fc3 = nn.Linear(N_bottleneck, N2)	# output = 1x392
		self.fc4 = nn.Linear(N2, N_output)		# output = 1x784
		self.type = 'NLP4'
		self.input_shape = (1, 28*28)

	def encode(self, X, Y, n):
		X = self.fc1(X)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)

		Y = self.fc1(Y)
		Y = F.relu(Y)
		Y = self.fc2(Y)
		Y = F.relu(Y)

		# imgs = [X,]
		# dont plot encoded/decoded original images?
		imgs = []
		for i in range(1, n + 1):
			img = torch.add(X * (n - i) / n, Y * i / n)
			imgs.append(img)
			# imgs.insert(-1, img)
		# imgs.append(Y)
		return imgs

	def decode(self, imgs):
		def _decode(img):
			img = self.fc3(img)
			img = F.relu(img)
			img = self.fc4(img)
			img = torch.sigmoid(img)
			return img
		# imgs = [_decode(x) for x in imgs]
		# dont plot encoded/decoded original images?
		for i in range(len(imgs)):
			# if i != 0 and i != len(imgs)-1:
			imgs[i] = _decode(imgs[i])
		return imgs

	def forward(self, X, Y, n):
		return self.decode(self.encode(X, Y, n))
