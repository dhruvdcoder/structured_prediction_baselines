import contextlib
import warnings
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Callable,
    Generator,
    overload,
    Iterator,
)

import numpy as np
import torch
from allennlp.common.lazy import Lazy
from allennlp.modules import FeedForward
from allennlp.training.optimizers import Optimizer
from structured_prediction_baselines.common import ModelMode
from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.task_nn import (
    TaskNN,
    CostAugmentedLayer,
)


@Sampler.register("inference-network")
class InferenceNetSampler(Sampler):
    def parameters_with_model_mode(
        self, mode: ModelMode
    ) -> Iterator[torch.nn.Parameter]:
        yield from self.inference_nn.parameters()
        if self.cost_augmented_layer is not None:
            yield from self.cost_augmented_layer.parameters()
        if self.tag_projection_layer is not None:
            yield from self.tag_projection_layer.parameters()

    def __init__(
        self,
        loss_fn: Loss,
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        tag_projection_layer: Optional[FeedForward] = None,
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )
        self.inference_nn = inference_nn
        self.cost_augmented_layer = cost_augmented_layer
        self.loss_fn = loss_fn
        self.tag_projection_layer = tag_projection_layer

        self.logging_children.append(self.loss_fn)

    @property
    def is_normalized(self) -> bool:
        """Whether the sampler produces normalized or unnormalized samples"""

        return False

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return y

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        y_hat, y_cost_aug = self._get_values(
            x, labels, buffer
        )  # (batch_size, 1, ...) Unnormalized

        if labels is not None:
            # compute loss for logging.
            loss = self.loss_fn(
                x,
                labels.unsqueeze(1),  # (batch, num_samples or 1, ...)
                y_hat,
                y_cost_aug,
                buffer,
            )
        else:
            loss = None

        return self.normalize(y_hat), self.normalize(y_cost_aug), loss

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.tag_projection_layer:
            y_inf: torch.Tensor = self.tag_projection_layer(
                self.inference_nn(x, buffer)
            ).unsqueeze(1)  # (batch_size, 1, ...) unormalized
        else:
            y_inf: torch.Tensor = self.inference_nn(x, buffer).unsqueeze(
                1
            )  # (batch_size, 1, ...) unormalized
        # inference_nn is TaskNN so it will output tensor of shape (batch, ...)
        # hence the unsqueeze

        if self.cost_augmented_layer is not None and labels is not None:
            y_cost_aug = self.cost_augmented_layer(
                torch.cat(
                    (
                        y_inf.squeeze(1),
                        labels.to(dtype=y_inf.dtype),
                    ),
                    dim=-1,
                ),
                buffer,
            ).unsqueeze(
                1
            )  # (batch_size,1, ...)
        else:
            y_cost_aug = None

        return y_inf, y_cost_aug


InferenceNetSampler.register("inference-network-unnormalized")(
    InferenceNetSampler
)

Sampler.register("inference-network-unnormalized")(InferenceNetSampler)
