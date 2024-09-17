from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score.structured_score import (
    StructuredScore,
)
import torch
import torch.nn as nn
import numpy as np
from structured_prediction_baselines.modules.task_nn import TaskNN


@StructuredScore.register("linear-chain")
class LinearChain(StructuredScore):
    def __init__(
        self, 
        num_tags: int, 
        task_nn:Optional[TaskNN] = None, 
        **kwargs: Any
        ):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__(task_nn = task_nn)
        self.M = 1
        self.num_tags = num_tags
        self.W = nn.Parameter(
            torch.FloatTensor(
                np.random.uniform(
                    -0.02, 0.02, (self.M, num_tags + 1, num_tags + 1)
                ).astype("float32")
            )
        )

    def forward(
        self,
        y: torch.Tensor,
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        mask = buffer.get("mask")
        batch_size, n_samples, seq_length, _ = y.shape
        targets = y.permute(
            2, 0, 1, 3
        )  # [seq_length, batch_size, n_samples, num_tags]
        length_index = mask.sum(1).long() - 1  # [batch_size]

        trans_energy: Union[float, torch.Tensor] = 0.0
        prev_labels = []

        for t in range(seq_length):
            energy_t = 0
            target_t = targets[t].float()  # [batch_size, n_samples, num_tags]

            if t < self.M:
                prev_labels.append(target_t.view(-1, self.num_tags))
                new_ta_energy = torch.mm(
                    prev_labels[t], self.W[t, -1, :-1].unsqueeze(dim=1)
                )
                new_ta_energy = new_ta_energy.squeeze(dim=-1)
                # [batch_size * n_samples, num_tags] x [num_tags, 1] -> [batch_size * n_samples]

                energy_t += new_ta_energy.view(batch_size, n_samples) * mask[:, t].unsqueeze(-1)
                # [batch_size, n_samples]

                for i in range(t):
                    new_ta_energy = torch.mm(
                        prev_labels[t - 1 - i], self.W[i, :-1, :-1]
                    )
                    # [batch_size * n_samples, num_tags]

                    energy_t += (
                        (
                            new_ta_energy.view(batch_size, n_samples, -1)
                            * target_t
                        ).sum(2)
                    ) * mask[:, t].unsqueeze(-1)
                    # [batch_size, n_samples]
            else:
                for i in range(self.M):
                    new_ta_energy = torch.mm(
                        prev_labels[self.M - 1 - i], self.W[i, :-1, :-1]
                    )
                    # [batch_size * n_samples, num_tags]

                    energy_t += (
                        (
                            new_ta_energy.view(batch_size, n_samples, -1)
                            * target_t
                        ).sum(2)
                    ) * mask[:, t].unsqueeze(-1)
                    # [batch_size, n_samples]

                prev_labels.append(target_t.view(-1, self.num_tags))
                prev_labels.pop(0)
            trans_energy += energy_t

        for i in range(min(self.M, seq_length)):
            pos_end_target = y[
                torch.arange(batch_size), :, length_index - i, :
            ].float()
            # [batch_size, n_samples, num_tags]
            pos_end_energy = torch.mm(
                pos_end_target.view(-1, self.num_tags),
                self.W[i, :-1, -1].unsqueeze(1),
            ).squeeze(1)
            trans_energy += pos_end_energy.view(batch_size, n_samples)
            # [batch_size, n_samples]

        return trans_energy
