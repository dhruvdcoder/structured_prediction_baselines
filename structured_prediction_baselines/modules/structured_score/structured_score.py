from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
from allennlp.common.lazy import Lazy
import torch
from structured_prediction_baselines.modules.task_nn import TaskNN


class StructuredScore(torch.nn.Module, Registrable):
    """Base class for all structured energy terms like linear-chain,
    skip-chain and other higher order energies.

    Inheriting classes should override the `foward` method.
    """
    def __init__(
        self,
        task_nn:Optional[TaskNN] = None
        ):
        super().__init__() 
        self.task_nn = task_nn
        
    def forward(
        self,
        y: torch.Tensor,
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            y: Tensor of shape (batch, num_samples or 1, ...)

        Returns:
            scores of shape (batch, num_samples or 1)
        """
        raise NotImplementedError

class StructuredScoreContainer(StructuredScore):
    """A collection of different `StructuredScore` modules
    that will be added together to form the total energy"""

    def __init__(self, constituent_energies: List[StructuredScore]) -> None:
        super().__init__()
        self.constituent_energies = torch.nn.ModuleList(constituent_energies)
        assert len(self.constituent_energies) > 0

    def forward(
        self,
        y: torch.Tensor,
        buffer: Dict,
        x: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if x is not None:
            total_energy: torch.Tensor = self.constituent_energies[0](
                y, buffer, x, **kwargs
            )
        else:
            total_energy: torch.Tensor = self.constituent_energies[0](
                y, buffer, **kwargs
            )

        for energy in self.constituent_energies[1:]:
            if x is not None:
                total_energy = total_energy + energy(y, buffer, x=x, **kwargs)
            else:
                total_energy = total_energy + energy(y, buffer, **kwargs)


        return total_energy
