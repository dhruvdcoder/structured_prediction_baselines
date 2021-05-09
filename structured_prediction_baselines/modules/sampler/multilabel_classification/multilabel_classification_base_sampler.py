from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.sampler import Sampler
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
        inference_nn: MultilabelTaskNN,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn,
            oracle_value_function,
        )
        self.inference_nn = inference_nn

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
            self.inference_nn(x, buffer=buffer).unsqueeze(1),
            None,
        )  # unormalized logits (batch, 1, ...)

    def get_metrics(self, reset: bool = False) -> dict:
        pass
