from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score import (
    StructuredScore,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
import torch
import math


@StructuredScore.register("multi-label-feedforward")
class MultilabelClassificationFeedforwardStructuredScore(StructuredScore):
    def __init__(
        self,
        feedforward: FeedForward,
    ):
        super().__init__()  # type:ignore
        self.feedforward = feedforward  # C1
        hidden_dim = self.feedforward.get_output_dim()  # type:ignore
        self.projection_vector = torch.nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim,))
        )  # c2 -> shape (hidden_dim,)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.feedforward(y.float())  # (batch, num_samples, hidden_dim)
        score = torch.nn.functional.linear(
            hidden, self.projection_vector
        )  # unormalized (batch, num_samples)

        return score
