from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score import (
    StructuredScore,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
# from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
# from allennlp.modules.seq2seq_encoders.multi_head_self_attention.MultiHeadSelfAttention
from allennlp.common.lazy import Lazy
from allennlp.nn.activations import Activation
from structured_prediction_baselines.modules.task_nn import TaskNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@StructuredScore.register("multi-label-feedforward")
class MultilabelClassificationFeedforwardStructuredScore(StructuredScore):
    def __init__(
        self,
        feedforward: FeedForward,
        task_nn: TaskNN
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward = feedforward
        hidden_dim = self.feedforward.get_output_dim()  # type:ignore
        self.projection_vector = torch.nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim,))
        )  # c2 -> shape (hidden_dim,)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.feedforward(y)  # (batch, num_samples, hidden_dim)
        score = torch.nn.functional.linear(
            hidden, self.projection_vector
        )  # unormalized (batch, num_samples)
        return score

@StructuredScore.register("multi-label-feedforward-global-v1")
class MultilabelClassificationFeedforwardStructuredScoreGlobalv1(StructuredScore):
    """
    E_Θglobal (x, y) = v^T MLP([y ; h_x] )
    where h_x = task_nn(x)
    """
    def __init__(
        self,
        feedforward1: FeedForward,
        feedforward2: FeedForward,
        task_nn: TaskNN
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward1 = feedforward1
        self.feedforward2 = feedforward2
        hidden_dim = self.feedforward2.get_output_dim()  # type:ignore
        self.projection_vector = torch.nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim,))
        )  # c2 -> shape (hidden_dim,)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        
        # x_representation.shape: (batch, D1)
        x_representation = self.task_nn(x, return_hidden_representation = True)        
        batch_size , x_hidden_dim = x_representation.shape

        # y.shape: (batch, num_samples, num_labels)
        batch_size, num_samples, num_labels = y.shape
        # y_representation.shape: (batch, N, D2)
        y_representation = self.feedforward1(y)

        # x_expanded_dim.shape: (batch, 1, D1)
        x_expanded_dim = x_representation.unsqueeze(1)
        # x_broadcasted.shape: (batch, num_samples, D1)
        x_broadcasted = x_expanded_dim.broadcast_to(batch_size, num_samples, x_hidden_dim)
        # x_concat_y.shape: (batch, num_samples, D1+D2)
        x_concat_y = torch.cat((x_broadcasted, y_representation),-1)

        hidden = self.feedforward2(x_concat_y)  # (batch, num_samples, hidden_dim)
        score = torch.nn.functional.linear(
            hidden, self.projection_vector
        )  # unormalized (batch, num_samples)

        return score

@StructuredScore.register("multi-label-feedforward-global-v2")
class MultilabelClassificationFeedforwardStructuredScoreGlobalv2(StructuredScore):
    """
    E_Θglobal (x, y) = v^T gamma(M_x y)
    where M_x = projection(task_nn(x))
    """
    def __init__(
        self,
        feedforward: FeedForward,
        task_nn: TaskNN,
        num_labels: int,
        activation: Activation
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward = feedforward
        self.num_labels = num_labels
        self.activation = activation
        self.hidden_dim = self.feedforward.get_output_dim() // num_labels
        self.projection_vector = torch.nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / self.hidden_dim), (self.hidden_dim,))
        )  # c2 -> shape (hidden_dim,)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        # x_representation.shape: (batch, D)
        x_representation = self.task_nn(x, return_hidden_representation = True)
        # y.shape : (batch, num_samples, num_labels)
        batch_size, num_samples, num_labels = y.shape
        # y_transposed.shape : (batch, L, num_samples)
        y_transposed = torch.transpose(y, 1,2)
        # M_x.shape: (batch, H, num_labels)
        M_x = self.feedforward(x_representation).view(batch_size, self.hidden_dim, num_labels)
        # Mx_y.shape: (batch, num_samples, H)
        Mx_y = torch.bmm(M_x, y_transposed).transpose(1,2)
        Mx_y = self.activation(Mx_y)
        # score.shape: (batch, num_samples)
        score = torch.nn.functional.linear(
            Mx_y, self.projection_vector
        )  # unormalized (batch, num_samples)
        return score

@StructuredScore.register("multi-label-feedforward-global-v3")
class MultilabelClassificationFeedforwardStructuredScoreGlobalv3(StructuredScore):
    """
    E_Θglobal (x, y) = v_x ^ T gamma(M y)
    where v_x = projection(task_nn(x))
    """
    def __init__(
        self,
        feedforward1: FeedForward,
        feedforward2: FeedForward,
        task_nn: TaskNN,
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward1 = feedforward1
        self.feedforward2 = feedforward2
        assert self.feedforward1.get_output_dim() == self.feedforward2.get_output_dim()
        self.hidden_dim = self.feedforward1.get_output_dim()

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        x_representation = self.feedforward2(self.task_nn(x, return_hidden_representation = True))
        batch_size, num_samples, num_labels = y.shape
        hidden = self.feedforward1(y)  # (batch, num_samples, hidden_dim)
        score = torch.sum(x_representation.unsqueeze(1) * hidden,2)
        # unormalized (batch, num_samples)
        return score

@StructuredScore.register("multi-label-feedforward-global-self-attention")
class MultilabelClassificationFeedforwardStructuredScoreGlobalSelfAttention(StructuredScore):
    """
    EΘglobal (x, y) = v^T (pool(self_attention(Hx, y1_emb, y2_emb, .. yL_emb)))
    where yi_emb = yi* active_embedding_matrix[i-1] + (1-yi)* passive_embedding_matrix[i-1]
    """
    def __init__(
        self,
        feedforward_x: FeedForward,
        feedforward_label1: FeedForward,
        feedforward_label2: FeedForward,
        task_nn: TaskNN,
        hidden_dim: int, 
        num_labels: int,
        num_heads: int,
        num_layers: int, 
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.num_labels = num_labels
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_dim = hidden_dim
        self.label_embeddings = torch.nn.Parameter(torch.empty((1, self.num_labels, self.hidden_dim)))
        nn.init.normal_(self.label_embeddings)
        self.feedforward_x = feedforward_x
        self.embedding_projection1 = feedforward_label1
        self.embedding_projection2 = feedforward_label2
        self.final_layer = torch.nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        # y.shape : (batch, num_samples, num_labels)
        batch_size, num_samples, num_labels = y.shape

        x_representation = self.task_nn(x, return_hidden_representation = True) # (batch, task_nn_D)
        _ , x_hidden_dim = x_representation.shape

        # now x is embedded into the same space as label embeddings
        x_representation = self.feedforward_x(x_representation) # (batch, D)

        # x_expanded_dim.shape: (batch, 1, D)
        x_expanded_dim = x_representation.unsqueeze(1)

        # x_broadcasted.shape: (batch, num_samples, D)
        x_broadcasted = x_expanded_dim.broadcast_to(batch_size, num_samples, self.hidden_dim)
        hx_reshaped = x_broadcasted.contiguous().view(batch_size*num_samples, 1, self.hidden_dim) # hx_reshaped.shape: (batch * num_samples,1, D)

        # embed y. Not adding position embeddings, as y embeddings are unique.
        y_reshaped = y.view(batch_size* num_samples, num_labels) # (batch_size*num_samples, num_labels)
        y_expanded_dim =  y_reshaped.unsqueeze(2) # (batch_size*num_samples, num_labels, 1) 
        active_embedding_matrix = self.embedding_projection1(self.label_embeddings) # (1, num_labels, hidden_dim)
        passive_embedding_matrix = self.embedding_projection2(self.label_embeddings) # (1, num_labels, hidden_dim)

        # (batch_size*num_samples, num_labels, hidden_dim)
        y_embedded = y_expanded_dim * active_embedding_matrix + (1-y_expanded_dim) * passive_embedding_matrix  
        
        # Concatenate x and y representations
        x_y_embedded = torch.cat((hx_reshaped, y_embedded),1)   # (batch_size * num_samples, num_labels + 1, hidden_dim)

        # Perform self attention without any masks.
        attention_output = self.transformer_encoder(x_y_embedded)[:,0,:]   # (batch_size*num_samples, hidden_dim)
        score_compact = self.final_layer(attention_output) # (batch_size*num_samples,1)
        score = score_compact.view(batch_size, num_samples) # (batch_size, num_samples)
        return score