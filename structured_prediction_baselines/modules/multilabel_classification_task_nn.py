from typing import List, Tuple, Union, Dict, Any, Optional
from .task_nn import TaskNN, CostAugmentedLayer, TextEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
import torch.nn as nn
import torch
import numpy as np


@TaskNN.register("multi-label-classification")
class MultilabelTaskNN(TaskNN):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_network: Union[FeedForward, TextEncoder],
        label_embeddings: Embedding,
    ):
        super().__init__()  # type:ignore
        self.feature_network: Union[FeedForward, TextEncoder] = feature_network
        self.label_embeddings = label_embeddings
        assert (
            self.label_embeddings.weight.shape[1]
            == self.feature_network.get_output_dim()  # type: ignore
        ), (
            f"label_embeddings dim ({self.label_embeddings.weight.shape[1]}) "
            f"and hidden_dim of feature_network"
            f" ({self.feature_network.get_output_dim()}) do not match."
        )

    def forward(
        self,
        x: torch.Tensor,
        buffer: Optional[Dict] = None,
        return_hidden_representation: bool = False, 
        **kwargs: Any,
    ) -> torch.Tensor:
        features = self.feature_network(x)  # (batch, hidden_dim)
        if return_hidden_representation:
            return features
        logits = torch.matmul(features, self.label_embeddings.weight.T)
        return logits  # unormalized logit of shape (batch, num_labels)


@TaskNN.register("multi-label-text-classification")
class MultilabelTextTaskNN(MultilabelTaskNN):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_network: TextEncoder,
        label_embeddings: Embedding,
    ):
        super().__init__(
            vocab=vocab,
            feature_network=feature_network,
            label_embeddings=label_embeddings,
        )


@CostAugmentedLayer.register("multi-label-stacked")
class MultiLabelStackedCostAugmentedLayer(CostAugmentedLayer):
    def __init__(
        self,
        feedforward: FeedForward,
        normalize_y: bool = True,
    ):
        super().__init__()
        self.feedforward = feedforward
        self.normalize_y = normalize_y

    def forward(self, yhat_y: torch.Tensor, buffer: Dict) -> torch.Tensor:
        """

        Args:
            yhat_y: Will be tensor of shape (batch, seq_len, 2*num_tags), where the first half of
                the last dimension will contain unnormalized yhat given by the test-time inference network.
                The second half will be ground truth labels.
        """
        assert (
            yhat_y.shape[-1] % 2 == 0
        ), "last dim of input to this layer should be 2*num_tags"
        num_tags = yhat_y.shape[-1] // 2

        if self.normalize_y:
            y_hat, y = torch.split(
                yhat_y, [num_tags, num_tags], dim=-1
            )  # type:ignore
            yhat_y = torch.cat((torch.sigmoid(y_hat), y), dim=-1)

        return self.feedforward(yhat_y)
