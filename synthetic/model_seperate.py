import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class net_x2(nn.Module):
	"""docstring for Relu_net"""
	def __init__(self):
		super(net_x2, self).__init__()
		self.linear_2_up   = nn.Linear(1,1,bias=False)
		self.linear_2_down = nn.Linear(1,1,bias=True)

	def forward(self,x1):
		x2=self.linear_2_up(x1)
		x2=F.sigmoid(x2)
		x2=self.linear_2_down(x2)
		return x2

class net_x3(nn.Module):
	"""docstring for Relu_net"""
	def __init__(self):
		super(net_x3, self).__init__()
		self.linear_3_up   = nn.Linear(2,1,bias=False)
		self.linear_3_down = nn.Linear(1,1,bias=True)

	def forward(self,x1,x2):
		x3=self.linear_3_up(torch.cat((x1,x2),dim=1))
		x3=F.sigmoid(x3)
		x3=self.linear_3_down(x3)
		return x3

class net_y(nn.Module):
	"""docstring for Relu_net"""
	def __init__(self):
		super(net_y, self).__init__()
		self.linear_y_up   = nn.Linear(3,1,bias=False)
		self.linear_y_down = nn.Linear(1,1,bias=True)

	def forward(self,x1,x2,x3):
		y=self.linear_y_up(torch.cat((x1,x2,x3),dim=1))
		y=F.sigmoid(y)
		y=self.linear_y_down(y)
		return y
