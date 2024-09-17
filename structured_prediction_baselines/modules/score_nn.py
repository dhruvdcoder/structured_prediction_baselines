from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
from allennlp.common.lazy import Lazy
import torch
from .task_nn import TaskNN
from .structured_score.structured_score import StructuredScore

class ScoreNN(torch.nn.Module, Registrable):
    """Concrete base class for creating feature representation for any task."""
    # default_implementation = 'global_only'
    def __init__(
        self,
        task_nn: TaskNN,  # (batch, ...)
        global_score: Optional[Lazy[StructuredScore]] = None,
        **kwargs: Any,
    ):
        super().__init__()  # type:ignore
        self.task_nn = task_nn
        if global_score is None:
            self.global_score = None
        else:
            self.global_score = global_score.construct(task_nn = self.task_nn)
        self._dtype = self.compute_input_dtype()

    def compute_input_dtype(self) -> Optional[torch.dtype]:
        if self.global_score is not None:
            for p in self.global_score.parameters():
                return p.dtype
        else:
            return None

        return None

    def compute_local_score(
        self, x: Any, y: Any, buffer: Dict, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        """
        Args:
            y: Will be of shape (batch, num_samples, ...).
                The task_nn will produce normalized score of shape (batch, ...).
                This function has to broadcast if required.
        """

        return None

    def compute_global_score(
        self, y: Any, buffer: Dict, x: Any=None, **kwargs: Any  #: (batch, num_samples, ...)
    ) -> Optional[torch.Tensor]:
        if self.global_score is not None:
            if x is not None:
                return self.global_score(y, buffer, x=x, **kwargs)
            else:
                return self.global_score(y, buffer, **kwargs)
        else:
            return None

    @property
    def input_dtype(self) -> Optional[torch.dtype]:
        return self._dtype

    def forward(
        self,
        x: Any,
        y: torch.Tensor,  # (batch, num_samples or 1, ...).
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        score = None
        local_score = self.compute_local_score(x, y, buffer, **kwargs)

        if local_score is not None:
            score = local_score
        
        global_score = self.compute_global_score(y, buffer, x=x, **kwargs)  # type: ignore

        if global_score is not None:
            if score is not None:
                score = score + global_score
            else:
                score = global_score

        return score  # (batch, num_samples)

ScoreNN.register("global_only")(ScoreNN)