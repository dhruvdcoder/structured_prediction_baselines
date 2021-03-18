from typing import List, Tuple, Union, Dict, Any, Optional, cast
from structured_prediction_baselines.modules.loss import Loss
from allennlp.common.checks import ConfigurationError
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
import torch


class DVNLoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

        if self.oracle_value_function is None:
            raise ConfigurationError(
                "oracle_value_function cannot be None for DVNLoss"
            )

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_probabilities: Optional[torch.Tensor],  # (batch, num_samples)
        **kwargs: Any,
    ) -> torch.Tensor:

        predicted_score, oracle_value = self._get_values(
            x, labels, y_hat, y_hat_probabilities, **kwargs
        )

        return self.compute_loss(predicted_score, oracle_value)

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_probabilities: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        num_samples = y_hat[1]
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect

        predicted_score = self.score_nn(
            x, y_hat, **kwargs
        )  # (batch, num_samples)

        if labels is not None:
            # For dvn we do not take gradient of oracle_score, so we detach y_hat
            oracle_score: Optional[torch.Tensor] = self.oracle_value_function(
                labels, y_hat.detach().clone(), **kwargs
            )  # (batch, num_samples)
        else:
            oracle_score = None

        return predicted_score, oracle_score


class DVNScoreLoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_probabilities: Optional[torch.Tensor],  # (batch, num_samples)
        **kwargs: Any,
    ) -> torch.Tensor:

        predicted_score = self._get_predicted_score(
            x, labels, y_hat, y_hat_probabilities, **kwargs
        )

        return self.compute_loss(predicted_score)

    def _get_predicted_score(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_probabilities: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect

        predicted_score = self.score_nn(
            x, y_hat, **kwargs
        )  # (batch, num_samples)

        return predicted_score
