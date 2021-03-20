from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch


class StructuredEnergy(torch.nn.Module, Registrable):
    """Base class for all structured energy terms like linear-chain,
    skip-chain and other higher order energies.

    Inheriting classes should override the `foward` method.
    """

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor = None,
        **kwargs: Any,
    ) -> Union[float, torch.Tensor]:
        raise NotImplementedError


class StructuredEnergyContainer(StructuredEnergy):
    """A collection of different `StructuredEnergy` modules
    that will be added together to form the total energy"""

    def __init__(self, constituent_energies: List[StructuredEnergy]) -> None:
        self.constituent_energies = torch.nn.ModuleList(constituent_energies)

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor = None,
        **kwargs: Any,
    ) -> Union[float, torch.Tensor]:
        total_energy: Union[float, torch.Tensor] = 0.0

        for energy in self.constituent_energies:
            total_energy = total_energy + energy(y, mask, **kwargs)

        return total_energy
