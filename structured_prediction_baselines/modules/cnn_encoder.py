from typing import Optional, Tuple

from overrides import overrides
import torch
from torch.nn import Conv2d, Linear, Dropout

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import min_value_of_dtype


@Seq2VecEncoder.register("cnn_2d")
class Cnn2dEncoder(Seq2VecEncoder):
    """
    Same as CnnEncoder, only difference is it uses 2D convolution layers instead of 1D
    # Parameters
    embedding_dim : `int`, required
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters : `int`, required
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes : `Tuple[int]`, optional (default=`(2, 3, 4, 5)`)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation : `Activation`, optional (default=`torch.nn.ReLU`)
        Activation to use after the convolution layers.
    output_dim : `Optional[int]`, optional (default=`None`)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is `None`, we will just return the result of the max pooling,
        giving an output of shape `len(ngram_filter_sizes) * num_filters`.
    """

    def __init__(
        self,
        num_tags: int,
        embedding_dim: int,
        num_filters: int,
        ngram_filter_sizes: Tuple[int, ...] = (3,),
        dropout: float = 0,
        conv_layer_activation: Activation = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self._embedding_dim = embedding_dim  # 1
        self._num_filters = num_filters  # 50
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or Activation.by_name("relu")()

        self._convolution_layers = [
            Conv2d(
                in_channels=self._embedding_dim,
                out_channels=self._num_filters,
                kernel_size=(ngram_size, num_tags),
                padding=(int(ngram_size / 2), 0)
            )
            for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)

        self._dropout = Dropout(dropout)
        # TODO:: Fix maxpool_output_dim, Add pooling?
        if output_dim:
            self.projection_layer = Linear(maxpool_output_dim, output_dim)
            self._output_dim = output_dim
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        if mask is not None:
            # mask: (batch, seq_len)
            tokens = tokens * mask.unsqueeze(1).unsqueeze(-1)

        batch_size, n_samples, seq_length, _ = tokens.shape
        tokens = tokens.reshape((batch_size*n_samples, 1, seq_length, -1))
        # input(tokens) shape: (batch_size, n_samples, seq_length, num_tags), we need to reshape and add a dimension to
        # match the dimensions that our convolution layer expects (N, in_channels, seq_length, num_tags)
        # here we have in_channels as 1

        filter_outputs = []

        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            # pool_length = tokens.shape[2] + 2 * convolution_layer.padding[0] - convolution_layer.kernel_size[0] + 1

            # Forward pass of the convolutions.
            # shape: (batch_size, num_filters, pool_length, 1)
            activations = self._activation(convolution_layer(tokens))

            filter_outputs.append(activations.squeeze(-1))

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters, pool_length)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers, pool_length)`.
        output = (
            torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        )

        output = self._dropout(output)
        return output.view(batch_size, n_samples, seq_length, -1)
