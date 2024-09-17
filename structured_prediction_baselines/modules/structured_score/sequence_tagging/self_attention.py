from typing import List, Tuple, Union, Dict, Any, Optional

from allennlp.nn.util import add_positional_features

from structured_prediction_baselines.modules.self_attention_encoder import SelfAttentionEncoder
from structured_prediction_baselines.modules.structured_score.structured_score import StructuredScore
import torch


@StructuredScore.register("self-attention")
class SelfAttention(StructuredScore):
    def __init__(self,
                 num_tags: int,
                 input_dim: int = None,
                 reduction: str = "max",
                 M: int = 0,
                 num_heads: int = 1,
                 attention_dim: int = None,
                 values_dim: int = None,
                 output_dim: int = None,
                 dropout: float = 0.1,
                 use_positional_encoding: bool = True,
                 **kwargs: Any):
        super().__init__()
        self.num_tags = num_tags
        self.reduction = reduction
        self.M = M
        assert self.M >= 0
        self.input_dim = input_dim or num_tags
        self.attention_dim = attention_dim or self.input_dim
        self.values_dim = values_dim or self.attention_dim
        self.attention_layer = SelfAttentionEncoder(
            num_heads,
            input_dim=self.input_dim,
            attention_dim=self.attention_dim,
            values_dim=self.values_dim,
            output_projection_dim=output_dim,
            attention_dropout_prob=dropout
        )
        self._use_positional_encoding = use_positional_encoding

    def forward(
        self,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        mask = buffer["mask"]
        batch_size, n_samples, seq_length, _ = y.shape
        attention_mask = self._get_attention_mask(n_samples, mask)

        # change the shape to appropriate structure for self-attention function.
        attention_input = y.view(batch_size * n_samples, seq_length, -1)
        if self._use_positional_encoding: # adding positional encoding.
            attention_input = add_positional_features(attention_input)


        attention_output = self.attention_layer(
            attention_input,
            attention_mask
        )  # (batch_size * n_samples, seq_length, num_tags)

        attention_output = attention_output.view(
            batch_size, n_samples, seq_length, -1
        ) * mask.unsqueeze(1).unsqueeze(-1)  # (batch_size, n_samples, seq_length, num_tags)

        if self.reduction == "sum":
            return attention_output.sum((2, 3))

        # reduction = "max" (Default)
        return attention_output.amax(dim=3).sum(2)

    def _get_attention_mask(self, n_samples, mask: torch.Tensor):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size, seq_length = mask.shape
        attention_mask = torch.BoolTensor(batch_size * n_samples, seq_length, seq_length).fill_(False)
        attention_mask = attention_mask.to(device=device)
        masked_length = mask.sum(dim=-1)

        for i in range(batch_size):
            i_seq_len = masked_length[i]
            for j in range(i_seq_len):
                lower_idx, higher_idx = max(0, j - self.M), min(i_seq_len, j + self.M + 1)
                attention_mask[i:i + n_samples, j, lower_idx:higher_idx] = True

        return attention_mask


@StructuredScore.register("self-attention-full-sequence")
class SelfAttentionFullSequence(SelfAttention):
    def __init__(self, num_tags: int,
                 reduction: str = "max",
                 **kwargs: Any):
        super().__init__(num_tags=num_tags, reduction=reduction, **kwargs)

    def _get_attention_mask(self, n_samples: int, mask: torch.Tensor):
        batch_size, seq_length = mask.shape
        attention_mask = mask.unsqueeze(1).expand(-1, seq_length, seq_length)
        attention_mask = attention_mask.repeat(1, n_samples, 1).view(batch_size * n_samples, seq_length, seq_length)
        return attention_mask

