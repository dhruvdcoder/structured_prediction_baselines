from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.sampler import (
    Sampler,
    SamplerModifier,
)
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.multilabel_classification_task_nn import (
    MultilabelTaskNN,
)


@Sampler.register("multi-label-basic")
class MultilabelClassificationSampler(Sampler):
    def __init__(
        self,
        task_nn: MultilabelTaskNN,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn,
            oracle_value_function,
        )
        self.task_nn = task_nn

    @property
    def is_normalized(self) -> bool:
        return False

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return (
            self.task_nn(x, buffer=buffer).unsqueeze(1),
            None,
        )  # unormalized logits (batch, 1, ...)


@SamplerModifier.register("multi-label-normalize")
class MultiLabelNormalize(SamplerModifier):
    def __call__(
        self, samples: torch.Tensor, samples_extra: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(samples)

    @property
    def is_normalized(self) -> bool:
        return True
