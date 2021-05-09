from typing import List, Tuple, Union, Dict, Any, Optional
from .sampler import Sampler
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
import torch


@Sampler.register("ground-truth")
class GroundTruthSampler(Sampler):
    """
    Returns ground truth labels as samples.

    It is obvious that for this sampler the supplied ground truth
    `labels` cannot be None.
    """

    known_dtypes = {"float": torch.float, "double": torch.double}

    def __init__(
        self,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        dtype: str = "float",
        **kwargs: Any,
    ):
        super().__init__(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )

        if dtype not in self.known_dtypes:
            raise ValueError(f"dtype should be one of {self.known_dtypes}")
        self.dtype = self.known_dtypes[dtype]

    @property
    def is_normalized(self) -> bool:
        """Whether the sampler produces normalized or unnormalized samples"""

        return True

    def forward(
        self, x: Any, labels: Any, buffer: Dict, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # NOTE: labels might be of dtype long
        # but samples should be float or double
        # we have no way of guessing that

        return (
            labels.unsqueeze(1).to(dtype=self.score_nn.input_dtype),
            None,
        )  # because labels will have shape (batch, ...)
        # and samples should have shape (batch, num_samples, ...)

    def get_metrics(self, reset: bool = False) -> dict:
        return self._metrics
