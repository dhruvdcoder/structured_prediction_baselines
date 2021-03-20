from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score.structured_score import StructuredScore
import torch
from allennlp_models.rc.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention


@StructuredScore.register("self-attention")
class SelfAttention(StructuredScore):
    def __init__(self, num_tags: int, reduction: str = "max", M: int = 0, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__()
        self.num_tags = num_tags
        self.reduction = reduction
        self.M = M
        # self.seq_length = seq_length
        self.attention_layer = MultiHeadSelfAttention(1, num_tags, num_tags, num_tags)

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size, n_samples, seq_length, _ = y.shape
        attention_mask = torch.BoolTensor(seq_length, seq_length).fill_(False)
        for i in range(seq_length):
            lower_idx, higher_idx = max(0, i - self.M), min(seq_length, i + self.M + 1)
            attention_mask[i][lower_idx:higher_idx] = True

        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_output = self.attention_layer(y, attention_mask)
        if self.reduction == "sum":
            return attention_output.sum((2, 3))

        # reduction = "max" (Default)
        return attention_mask.amax(dim=3).sum(2)