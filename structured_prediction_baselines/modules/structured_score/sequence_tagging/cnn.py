from typing import List, Tuple, Union, Dict, Any, Optional

from allennlp.nn import Activation

from structured_prediction_baselines.modules.structured_score.structured_score import StructuredScore
import torch
from structured_prediction_baselines.modules.cnn_encoder import Cnn2dEncoder


@StructuredScore.register("cnn")
class CNN(StructuredScore):

    def __init__(self,
                 num_tags: int,
                 embedding_dim: int,
                 num_filters: int,
                 ngram_filter_sizes: Tuple[int, ...],
                 dropout: float = 0.1,
                 conv_layer_activation: Activation = None,
                 output_dim: Optional[int] = None,
                 **kwargs: Any):
        super().__init__()
        self.num_tags = num_tags
        self.encoder = Cnn2dEncoder(num_tags=num_tags,
                                    embedding_dim=embedding_dim,
                                    num_filters=num_filters,
                                    ngram_filter_sizes=ngram_filter_sizes,
                                    dropout=dropout,
                                    conv_layer_activation=conv_layer_activation,
                                    output_dim=output_dim)

    def forward(
        self,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        mask = buffer["mask"]  # (batch, seq_len)
        output = self.encoder(y, mask)  # (batch_size, num_samples or 1, ...)
        output = output.sum(dim=-1)  # (batch_size, num_samples or 1, seq_length)
        output = output * mask.unsqueeze(1)
        return output.sum(dim=2)
