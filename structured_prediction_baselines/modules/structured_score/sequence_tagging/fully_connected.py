from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score.structured_score import StructuredScore
from structured_prediction_baselines.modules.task_nn import TaskNN
import torch

@StructuredScore.register("fully_connected")
class FullyConnected(StructuredScore):
    def __init__(
        self, 
        task_nn:Optional[TaskNN] = None,
        **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__(task_nn=task_nn)
        # TODO: initialize weights
        # self.W =

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor = None,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        # implement
        pass
