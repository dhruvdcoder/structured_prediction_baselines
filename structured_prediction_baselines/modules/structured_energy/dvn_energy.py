from typing import List, Tuple, Union, Dict, Any, Optional
from structured_energy import StructuredEnergy
import torch.nn as nn
import torch
import numpy as np

@StructuredEnergy.register("dvn")
class DVN(StructuredEnergy):
	def __init__(self, label_dim, num_pairwise, non_linearity=nn.Softplus()):
		super().__init__()
		self.non_linearity = non_linearity
		self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
		torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

		self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
		torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

	def forward(self, y: torch.Tensor, 
				y_hat: torch.Tensor, 
				mask: torch.BoolTensor = None) -> Union[float, torch.Tensor]:
		y_hat = torch.mul(y, y_hat)
		y_hat = torch.sum(y_hat, dim=1)
		y_hat = y_hat.view(y_hat.size()[0], 1)

		e_label = self.non_linearity(torch.mm(y, self.C1))
		e_label = torch.mm(e_label, self.c2)
		e_global = torch.add(e_label, y_hat)

		return e_global

if __name__ == '__main__':
	model = DVN(label_dim=10, num_pairwise = 10, non_linearity=nn.Softplus())

	a = torch.Tensor([[-0.1860,  1.0739, -0.1437,  1.5116,  0.3117,  0.4322,  1.5382,  1.6908,
        -1.0380,  0.2666]])
	print(model(a,a))