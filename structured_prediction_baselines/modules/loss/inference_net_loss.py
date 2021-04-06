from typing import Any, Optional, Tuple, cast, Union, Dict, Callable
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn import util
from torch.nn.functional import relu
import torch.nn.functional as F

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN


class MarginBasedLoss(Loss):
    """
    Implements the losses described in the SPEN+Inference Network papers.

    We compute $Delta(y_hat, y^*)$ as (oracle_score(y^*) - oracle_score(y_hat)).

    1. Margin Rescaled Loss:

    """

    margin_types: Dict[
        str,
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
    ] = {
        "margin-rescaled-zero-truncation": (
            lambda oracle_cost, cost_augmented_inference_score, ground_truth_score: torch.relu(
                oracle_cost
                - (ground_truth_score - cost_augmented_inference_score)
            )
        ),
        "slack-rescaled-zero-truncation": (
            lambda oracle_cost, cost_augmented_inference_score, ground_truth_score: (
                oracle_cost
                * torch.relu(
                    1.0 - (ground_truth_score - cost_augmented_inference_score)
                )
            )
        ),
        "perceptron-zero-truncation": (
            lambda oracle_cost, cost_augmented_inference_score, ground_truth_score: torch.relu(
                cost_augmented_inference_score - ground_truth_score
            )
        ),
        "contrastive-zero-truncation": (
            lambda oracle_cost, cost_augmented_inference_score, ground_truth_score: torch.relu(
                1.0 - (ground_truth_score - cost_augmented_inference_score)
            )
        ),
    }

    def __init__(
        self,
        score_nn: ScoreNN,
        oracle_value_function: OracleValueFunction,
        reduction: Optional[str] = "none",
        normalize_y: bool = True,
        margin_type: str = "margin-rescaled-zero-truncation",
        perceptron_loss_weight: float = 0.0,
        **kwargs: Any,
    ):
        """
        margin_type: one of [''] as described here https://arxiv.org/pdf/1803.03376.pdf
        cross_entropy: set True when training inference network and false for energy function, default True
        zero_truncation: set True when training for energy function, default False

        """
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            reduction=reduction,
            normalize_y=normalize_y,
        )

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None")

        if self.oracle_value_function is None:
            raise ConfigurationError("oracle_value_function cannot be None")

        if margin_type not in self.margin_types:
            raise ConfigurationError(
                f"margin_type must be one of {self.margin_types}"
            )
        self.margin_type = margin_type
        self.perceptron_loss_weight = perceptron_loss_weight

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...) unnormalized
        y_hat_extra: Optional[
            torch.Tensor
        ],  # (batch, num_samples, ...), unnormalized
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        y_inf = y_hat
        y_cost_aug = y_hat_extra
        assert buffer is not None
        (
            oracle_cost,
            cost_aug_score,
            inference_score,
            ground_truth_score,
        ) = self._get_values(x, labels, y_inf, y_cost_aug, buffer)
        loss_unreduced = self.margin_types[self.margin_type](
            oracle_cost, cost_aug_score, ground_truth_score
        )

        if self.perceptron_loss_weight:
            loss_unreduced = loss_unreduced + self.perceptron_loss_weight * (
                torch.relu(inference_score - ground_truth_score)
            )

        return loss_unreduced

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,  # Assumed to be unnormalized
        y_cost_aug: Optional[torch.Tensor],  # assumed to be unnormalized
        buffer: dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        assert (
            labels is not None
        )  # if you call this loss, labels cannot be None

        if y_cost_aug is None:
            y_cost_aug = (
                y_hat if not self.normalize_y else self.normalize(y_hat)
            )

        if self.normalize_y:
            y_hat = self.normalize(y_hat)
        ground_truth_score = self.score_nn(
            x, labels.to(dtype=y_hat.dtype), buffer
        )
        inference_score = self.score_nn(x, y_hat, buffer)
        cost_aug_score = self.score_nn(x, y_cost_aug, buffer)

        oracle_cost: torch.Tensor = self.oracle_value_function.compute_as_cost(
            labels, y_cost_aug, mask=buffer.get("mask")
        )  # (batch, num_samples)

        return (
            oracle_cost,
            cost_aug_score,
            inference_score,
            ground_truth_score,
        )


class InferenceLoss(MarginBasedLoss):
    """
    The class exclusively outputs loss (lower the better) to train the paramters of the inference net.
    Last equation in the section 2.3 in "An Exploration of Arbitrary-Order Sequence Labeling via Energy-Based Inference Networks".

    Note:
        Right now we always drop zero truncation.
    """

    def __init__(self, inference_score_weight: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.inference_score_weight = inference_score_weight

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...) might be unnormalized
        y_hat_extra: Optional[
            torch.Tensor
        ],  # (batch, num_samples, ...), might be unnormalized
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        y_inf = y_hat
        y_cost_aug = y_hat_extra
        assert buffer is not None
        (
            oracle_cost,
            cost_augmented_inference_score,
            inference_score,
            ground_truth_score,
        ) = self._get_values(x, labels, y_inf, y_cost_aug, buffer)
        loss_unreduced = -(
            oracle_cost
            + cost_augmented_inference_score - ground_truth_score
            + self.inference_score_weight * (inference_score - ground_truth_score)
        )  # the minus sign turns this into argmin objective

        return loss_unreduced
