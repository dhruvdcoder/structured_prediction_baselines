from task_nn import TaskNN
import torch.nn as nn
import torch
import numpy as np


@TaskNN.register("multi-label-dvn")
class MultilabelTaskNN(TaskNN):
	def __init__(self, feature_dim, label_dim, num_hidden, 
				num_pairwise, add_second_layer = True, non_linearity=nn.Softplus()):
		super().__init__()
		self.non_linearity = non_linearity
		self.fc1 = nn.Linear(feature_dim, num_hidden)
		if add_second_layer:
			self.fc2 = nn.Linear(num_hidden, num_hidden)
			# Add what corresponds to the b term in SPEN
			# Eq. 4 in http://www.jmlr.org/proceedings/papers/v48/belanger16.pdf
			# X is Batch_size x num_hidden
			# B is num_hidden x L, so XB gives a Batch_size x L label
			# and then we multiply by the label and sum over the L labels,
			# and we get Batch_size x 1 output  for the local energy
			self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
			torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
		else:
			self.fc2 = nn.Linear(num_hidden, label_dim)
			self.B = None
		# Label energy terms, C1/c2  in equation 5 of SPEN paper
		self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
		torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))
		self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
		torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

	def forward(self, x, y):
		x = self.non_linearity(self.fc1(x))
		x = self.non_linearity(self.fc2(x))
		print(self.B)
		if self.B is not None:
			e_local = torch.matmul(x, self.B)
		else:
			e_local = x
		return e_local

if __name__ == '__main__':
	model = MultilabelTaskNN(feature_dim = 10, label_dim = 10, num_hidden = 10, 
				num_pairwise = 10, add_second_layer = True, non_linearity=nn.Softplus())
	a = torch.ones(10)
	print(model(a, a))
	print("?")


