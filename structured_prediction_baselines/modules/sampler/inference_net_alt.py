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
)

import numpy as np
import torch
from allennlp.common.lazy import Lazy
from allennlp.training.optimizers import Optimizer

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.stopping_criteria import (
    StoppingCriteria,
    StopAfterNumberOfSteps,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.task_nn import (
    TaskNN,
    CostAugmentedLayer,
)


@Sampler.register("inference-network-alt", constructor="from_partial_objects")
class InferenceNetSampler(Sampler):
    def __init__(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        name: str = 'inf_net',
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            score_nn,
            oracle_value_function,
            name
        )
        self.inference_nn = inference_nn
        self.cost_augmented_layer = cost_augmented_layer
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if isinstance(stopping_criteria, int):
            self.stopping_criteria: StoppingCriteria = StopAfterNumberOfSteps(
                stopping_criteria
            )
        else:
            self.stopping_criteria = stopping_criteria
        self._eval_grad = False

    @property
    def is_normalized(self) -> bool:
        """Whether the sampler produces normalized or unnormalized samples"""

        return False

    @classmethod
    def from_partial_objects(
        cls,
        optimizer: Lazy[Optimizer],
        loss_fn: Lazy[Loss],  #: This loss can be different from the main loss
        inference_nn: TaskNN,  #: inference_nn cannot be None for this sampler
        score_nn: ScoreNN,
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        name: str = 'inf_net',
    ) -> "InferenceNetSampler":
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        trainable_parameters: Dict[str, torch.Tensor] = {}

        trainable_parameters.update(inference_nn.named_parameters())

        if cost_augmented_layer is not None:
            trainable_parameters.update(
                cost_augmented_layer.named_parameters()
            )
        optimizer_ = optimizer.construct(
            model_parameters=list(trainable_parameters.items())
        )

        return cls(
            optimizer_,
            loss_fn_,
            inference_nn=inference_nn,
            score_nn=score_nn,
            cost_augmented_layer=cost_augmented_layer,
            oracle_value_function=oracle_value_function,
            stopping_criteria=stopping_criteria,
            name=name
        )

    def get_loss_fn(
        self, x: Any, labels: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor]:

        if self.training and (labels is None):
            warnings.warn("Labels should not be None in training mode!")

        def loss_fn(
            y_hat: torch.Tensor, y_cost_aug: torch.Tensor, buffer: Dict
        ) -> torch.Tensor:
            return self.loss_fn(
                x,
                labels,
                y_hat,
                y_cost_aug,
                buffer,
            )

        return loss_fn

    def get_dtype_device(self) -> Tuple[torch.dtype, torch.device]:
        for param in self.loss_fn.parameters():
            dtype = param.dtype
            device = param.device

            break

        return dtype, device

    def get_batch_size(self, x: Any) -> int:
        if isinstance(x, torch.Tensor):
            return x.shape[0]
        else:
            raise NotImplementedError

    @contextlib.contextmanager
    def only_inference_nn_grad_on(self) -> Generator[None, None, None]:

        trainable_parameters = [
            p for g in self.optimizer.param_groups for p in g["params"]
        ]
        # switch off the gradients for score_nn but first cache their requires grad
        assert self.score_nn is not None
        score_nn_requires_grad_map = {
            name: param.requires_grad
            for name, param in self.score_nn.named_parameters()
        }
        try:
            # first switch off all.

            for p in self.score_nn.parameters():
                p.requires_grad_(False)
            # then switch on inf net params

            for g in self.optimizer.param_groups:
                for p in g["params"]:
                    p.requires_grad_(True)
            yield
        finally:
            # set the requires_grad back to false for inf net

            for n, p in self.score_nn.named_parameters():  # type: ignore
                p.requires_grad_(score_nn_requires_grad_map[n])

            for g in self.optimizer.param_groups:
                for p in g["params"]:
                    p.requires_grad_(False)

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if labels is None or (not self.training):
            y_inf: torch.Tensor = self.inference_nn(x, buffer).unsqueeze(
                1
            )  # (batch_size, 1, ...)

            return y_inf, None
        else:
            # switch on gradients on the parameters of inference network using context manager
            with self.only_inference_nn_grad_on():
                labels = labels.unsqueeze(1)
                loss_fn = self.get_loss_fn(
                    x, labels
                )  #: Loss function will expect labels in form (batch, num_samples or 1, ...)

                loss_values: List[float] = []
                step_number = 0
                loss_value: Union[torch.Tensor, float] = float("inf")

                if self._eval_grad:
                    while not self.stopping_criteria(
                        step_number, float(loss_value)
                    ):
                        self.optimizer.zero_grad(set_to_none=True)
                        y_inf, y_cost_aug = self._get_values(x, labels, buffer)
                        loss_value = self.update(
                            y_inf, y_cost_aug, buffer, loss_fn
                        )

                        loss_values.append(float(loss_value))

                        step_number += 1
                    self._metrics['inf_net_loss'] = np.mean(loss_values)
                    self._total_loss += np.mean(loss_values)
                    self._num_batches += 1
                    self._eval_grad = False
                else:
                    y_inf, y_cost_aug = self._get_values(x, labels, buffer)
                    self._eval_grad = True

            # once out of Sampler, y_inf and y_cost_aug should not get gradients
            return (
                y_inf.detach().clone(),
                y_cost_aug.detach().clone()
                if y_cost_aug is not None
                else None,
            )

    def update(
        self,
        y_hat: torch.Tensor,
        y_cost_aug: torch.Tensor,
        buffer: Dict,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns:
            loss: loss value at the previous point (unreduced)
        """
        loss = loss_fn(y_hat, y_cost_aug, buffer)

        loss.backward()  # type:ignore
        self.optimizer.step()

        return loss

    def _get_values(
        self,
        x: Any,
        labels: torch.Tensor,
        buffer: Dict,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        y_inf: torch.Tensor = self.inference_nn(
            x, buffer
        ).unsqueeze(
            1
        )  # (batch_size, 1, ...) unormalized
        # inference_nn is TaskNN so it will output tensor of shape (batch, ...)
        # hence the unsqueeze

        if self.cost_augmented_layer is not None:
            y_cost_aug = self.cost_augmented_layer(
                torch.cat(
                    (
                        y_inf.squeeze(1),
                        labels.to(dtype=y_inf.dtype).squeeze(1),
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

    def get_metrics(self, reset: bool = False) -> dict:
        metrics = self._metrics
        metrics['total_' + self.name + '_loss'] = float(
            self._total_loss / self._num_batches) if self._num_batches > 0 else 0.0
        if reset:
            self._metrics = {}
            self._total_loss = 0.0
            self._num_batches = 0
            metrics.pop(self.name + '_loss', None)
        else:
            loss_metrics = self.loss_fn.get_metrics(reset=True)
            for metric in loss_metrics:
                loss_metrics[metric] /= self.stopping_criteria.number_of_steps
            metrics.update(loss_metrics)

        return metrics
