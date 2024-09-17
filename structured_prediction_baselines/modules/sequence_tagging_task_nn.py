import logging

from .task_nn import TaskNN, CostAugmentedLayer
from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import (
    Seq2SeqEncoder,
    TimeDistributed,
    TextFieldEmbedder,
    FeedForward,
)
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
import allennlp.nn.util as util

logger = logging.getLogger(__name__)


@TaskNN.register("sequence-tagging")
class SequenceTaggingTaskNN(TaskNN):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = 0,
        label_namespace: str = "labels",
        project_onto_tags: bool = True

    ):
        """

        Args:
            text_field_embedder : `TextFieldEmbedder`, required
                Used to embed the tokens `TextField` we get as input to the model.
            encoder : `Seq2SeqEncoder`
                The encoder that we will use in between embedding tokens and predicting output tags.
            feedforward : `FeedForward`, optional, (default = `None`).
                An optional feedforward layer to apply after the encoder.

        """
        super().__init__()  # type:ignore
        self.num_tags = vocab.get_vocab_size(namespace=label_namespace)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.feedforward = feedforward
        output_dim = self.text_field_embedder.get_output_dim()

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()  # type: ignore
        elif self.encoder is not None:
            output_dim = self.encoder.get_output_dim()

        if output_dim is None:
            raise ValueError("output_dim cannot be None")

        if project_onto_tags:
            self.tag_projection_layer = TimeDistributed(
                Linear(output_dim, self.num_tags)
            )  # equivalent to Uj.b(x,t) in eq (3)
        else:
            self.tag_projection_layer = None
            logger.warning("Tag Projection Layer set to None. TaskNN would return encoded text instead of logits.")
        if dropout:
            self.dropout: Optional[torch.nn.Module] = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        buffer: Dict,
    ) -> torch.Tensor:
        mask = buffer.get("mask")

        if mask is None:
            mask = util.get_text_field_mask(tokens)
        buffer["mask"] = mask

        embedded_text_input = self.text_field_embedder(tokens)
        buffer["embedded_x"] = embedded_text_input
        if self.encoder:
            encoded_text = self.encoder(embedded_text_input, mask)
        else:
            encoded_text = embedded_text_input

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self.feedforward:
            encoded_text = self.feedforward(encoded_text)

        if self.tag_projection_layer:
            logits = self.tag_projection_layer(encoded_text)

            return (
                logits  # shape (batch, sequence, num_tags) of unormalized logits
            )

        return encoded_text  # shape (batch, sequence, output_dim)


@CostAugmentedLayer.register("sequence-tagging-stacked")
class SequenceTaggingStackedCostAugmentedLayer(CostAugmentedLayer):
    def __init__(
        self,
        seq2seq: Seq2SeqEncoder,
        normalize_y: bool = True,
    ):
        super().__init__()
        self.seq2seq = seq2seq
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
            y_hat, y = torch.split(yhat_y, [num_tags, num_tags], dim=-1)
            yhat_y = torch.cat((torch.softmax(y_hat, dim=-1), y), dim=-1)
        mask = buffer.get("mask")
        assert mask is not None

        return self.seq2seq(yhat_y, mask=mask)
