from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score.structured_score import StructuredScore
import torch
import torch.nn as nn
import numpy as np
from structured_prediction_baselines.modules.task_nn import TaskNN

@StructuredScore.register("vkp")
class VKP(StructuredScore):
    """Vectorized Kronecker Product High Order Energy"""

    def __init__(self, num_tags: int, M: int, task_nn:Optional[TaskNN]=None, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__(task_nn = task_nn)
        self.num_tags = num_tags
        self.M = M
        self.W = nn.Parameter(
            torch.FloatTensor(np.random.uniform(-0.02, 0.02, [num_tags + 1] * (self.M + 1)).astype('float32')))

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor = None,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        B, T, C = y.shape
        targets = y.transpose(0, 1)
        Wtrans0 = _get_start_trans(self.W, self.num_tags, self.M)  # [C+1]
        Wtrans0 = _get_trans_without_eos_sos(Wtrans0, self.num_tags, 0)  # [C]
        energy_0 = torch.mm(targets[0], Wtrans0.unsqueeze(1)).squeeze(1)  # [B]
        trans_energy = energy_0

        trans_no_eos = _get_trans_without_eos_sos(self.W, self.num_tags,
                                                  self.M)  # [C]*(M+1) transition matrix without <sos> or <eos>
        # print("trans_no_eos {}".format(trans_no_eos.size()))
        # print("M {}".format(self.M))
        trans_no_eos = trans_no_eos.reshape((-1, C))  # [C^M, C]
        # print("trans_no_eos {}".format(trans_no_eos.size()))
        for t in range(1, T):
            energy_t = 0
            target_t = targets[t]  # [B, C]
            if t < self.M:
                Wtrans = _get_start_trans(self.W, self.num_tags, self.M - t)  # [C+1]*(t+1)
                Wtrans = _get_trans_without_eos_sos(Wtrans, self.num_tags, t)  # [C]*(t+1)
                Wtrans = Wtrans.reshape((-1, C))  # [C^t, C]
                com_targets = targets[0]  # [B, C]
                for i in range(1, t):
                    # [B, C^i, 1] x [B, 1, C] -> [B, C^i, C]
                    com_targets = torch.matmul(com_targets.unsqueeze(2), targets[i].unsqueeze(1))
                    com_targets = com_targets.reshape((B, -1))  # [B, C^(i+1)]
                new_ta_energy = torch.mm(com_targets, Wtrans)  # [B, C^t] x [C^t, C] -> [B, C]
                energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [B]
            else:
                com_targets = targets[t - self.M]  # [B, C]
                for i in range(t - self.M + 1, t):
                    # [B, C^i, 1] x [B, 1, C] -> [B, C^i, C]
                    com_targets = torch.matmul(com_targets.unsqueeze(2), targets[i].unsqueeze(1))
                    com_targets = com_targets.reshape((B, -1))  # [B, C^(i+1)]
                # print("com_targets : {}".format(com_targets.size()))
                new_ta_energy = torch.mm(com_targets, trans_no_eos)  # [B, C^M] x [C^M, C] -> [B, C]
                # print("new_ta_energy : {}".format(new_ta_energy.size()))
                energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [B]
            trans_energy += energy_t

            return trans_energy


def _get_end_trans(trans, K, M, maxM):
    assert M <= maxM
    if M == 1:
        if maxM == 1:
            slps = trans[:, K]
        elif maxM == 2:
            slps = trans[:, :, K]
        elif maxM == 3:
            slps = trans[:, :, :, K]
        elif maxM == 4:
            slps = trans[:, :, :, :, K]
    elif M == 2:
        slps = trans[K, K]
        if maxM == 2:
            slps = trans[:, K, K]
        elif maxM == 3:
            slps = trans[:, :, K, K]
        elif maxM == 4:
            slps = trans[:, :, :, K, K]
    elif M == 3:
        slps = trans[K, K, K]
        if maxM == 3:
            slps = trans[:, K, K, K]
        elif maxM == 4:
            slps = trans[:, :, K, K, K]
    elif M == 4:
        if maxM == 4:
            slps = trans[:, K, K, K, K]
    return slps


def _get_start_trans(trans, K, M):
    if M == 1:
        slps = trans[K]
    elif M == 2:
        slps = trans[K, K]
    elif M == 3:
        slps = trans[K, K, K]
    elif M == 4:
        slps = trans[K, K, K, K]
    return slps


def _get_trans_without_eos_sos(trans, K, M):
    if M == 0:
        tlps = trans[:K]
    elif M == 1:
        tlps = trans[:K, :K]
    elif M == 2:
        tlps = trans[:K, :K, :K]
    elif M == 3:
        tlps = trans[:K, :K, :K, :K]
    elif M == 4:
        tlps = trans[:K, :K, :K, :K, :K]
    return tlps