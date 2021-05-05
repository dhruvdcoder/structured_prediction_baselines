from task_nn import TaskNN
import torch.nn as nn
import torch
import numpy as np


@TaskNN.register("multi-label-spen")
class MultilabelTaskNN(TaskNN):
	def __init__(self, weights_last_layer_mlp, feature_dim=150, label_dim=159,
                 num_pairwise=16, non_linearity=nn.Softplus()):
		super().__init__()
		self.non_linearity = non_linearity
		self.B = torch.nn.Parameter(torch.transpose(-weights_last_layer_mlp, 0, 1))

		self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
		torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

		self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
		torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

	def forward(self, x, y):
		return torch.matmul(x, self.B)
