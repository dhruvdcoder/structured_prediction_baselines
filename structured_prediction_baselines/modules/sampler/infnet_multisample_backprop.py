import contextlib
import warnings
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Callable,
    Generator,
)

import torch
from allennlp.common.lazy import Lazy
from allennlp.training.optimizers import Optimizer

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.stopping_criteria import (
    StoppingCriteria,
    StopAfterNumberOfSteps,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.task_nn import (
    TaskNN,
    CostAugmentedLayer,
)


# keep prob = True, and num_samples=0 should return the same model with infnet + DVN.
@Sampler.register("infnet-multi-sample-learner", constructor="from_partial_objects")
# --> change it to "multi-sample-logprob"
class InfnetMultiSampleLearner(Sampler):
    """
    Implements the inference network that trains infnet (Task-NN) with multiple samples.
    Draw discrete samples (s_i) from Task-NN output (y_hat) and then evaluate their gradients (g) on Score-NN
    loss_function: g * binary_cross_entropy(labels=s_i,y_hat)  
    """
    def __init__(
        self,
        optimizer: Optimizer, #loss_fn: Loss,  
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Loss,
        loss_fn_sample: Optional[Loss] = None, 
        loss_fn_for_grad: Optional[Loss] = None,
        sample_loss_weight: Optional[float] = 1.0,
        num_samples: int = 1,
        keep_probs: bool = False, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            score_nn,
            oracle_value_function,
        )
        self.inference_nn = inference_nn
        self.cost_augmented_layer = cost_augmented_layer
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if isinstance(stopping_criteria, int):
            self.stopping_criteria: StoppingCriteria = StopAfterNumberOfSteps(
                stopping_criteria
            )
        else:
            self.stopping_criteria = stopping_criteria

        # Till here: the same as inference_net.py  
        # After this point: new variables for InfnetMultiSampleLearner
        self.loss_fn_sample = loss_fn_sample
        self.loss_fn_for_grad = loss_fn_for_grad
        self.sample_loss_weight = sample_loss_weight
        self.num_samples = num_samples
        self.keep_probs = keep_probs

    ## copied from "InferenceNetSampler"
    @property
    def is_normalized(self) -> bool:
        """Whether the sampler produces normalized or unnormalized samples"""

        return False

    @classmethod
    def from_partial_objects(
        cls,
        optimizer: Lazy[Optimizer],
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Lazy[Loss], # first loss is always multi-sample loss.
        loss_fn_sample: Lazy[Loss], 
        loss_fn_for_grad: Lazy[Loss],
        sample_loss_weight: Optional[float] = 1.0,
        num_samples: int = 1,
        keep_probs: bool = False, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ) -> "InfnetMultiSampleLearner":
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        loss_fn_sample_ = loss_fn_sample.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        loss_fn_for_grad_ = loss_fn_for_grad.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        
        trainable_parameters: Dict[str, torch.Tensor] = {}

        trainable_parameters.update(inference_nn.named_parameters())

        if cost_augmented_layer is not None:
            trainable_parameters.update(
                cost_augmented_layer.named_parameters()
            )
        optimizer_ = optimizer.construct(
            model_parameters=list(trainable_parameters.items())
        )

        return cls(
            optimizer=optimizer_,
            inference_nn=inference_nn,
            score_nn=score_nn,
            loss_fn=loss_fn_,
            loss_fn_sample=loss_fn_sample_,
            loss_fn_for_grad=loss_fn_for_grad_,
            sample_loss_weight=sample_loss_weight,
            num_samples=num_samples,
            keep_probs=keep_probs,
            cost_augmented_layer=cost_augmented_layer,
            oracle_value_function=oracle_value_function,
            stopping_criteria=stopping_criteria,
        )
    
    def get_dtype_device(self) -> Tuple[torch.dtype, torch.device]:
        for param in self.loss_fn.parameters():
            dtype = param.dtype
            device = param.device

            break

        return dtype, device

    def get_batch_size(self, x: Any) -> int:
        if isinstance(x, torch.Tensor):
            return x.shape[0]
        else:
            raise NotImplementedError

    @contextlib.contextmanager
    def only_inference_nn_grad_on(self) -> Generator[None, None, None]:

        trainable_parameters = [
            p for g in self.optimizer.param_groups for p in g["params"]
        ]
        # switch off the gradients for score_nn but first cache their requires grad
        assert self.score_nn is not None
        score_nn_requires_grad_map = {
            name: param.requires_grad
            for name, param in self.score_nn.named_parameters()
        }
        try:
            # first switch off all.

            for p in self.score_nn.parameters():
                p.requires_grad_(False)
            # then switch on inf net params

            for g in self.optimizer.param_groups:
                for p in g["params"]:
                    p.requires_grad_(True)
            yield
        finally:
            # set the requires_grad back to false for inf net

            for n, p in self.score_nn.named_parameters():  # type: ignore
                p.requires_grad_(score_nn_requires_grad_map[n])

            for g in self.optimizer.param_groups:
                for p in g["params"]:
                    p.requires_grad_(False)

    def draw_samples(self, y_inf):
        assert y_inf.dim() == 3
        assert y_inf.shape[1] == 1
        y_hat_n = torch.sigmoid(y_inf)
        ## commented out as we won't really sample from cost_aug layer.
        # y_hat_extra_n = ( 
        #     torch.sigmoid(y_cost_aug) if y_cost_aug is not None else None
        # )                    
        if self.training and self.num_samples>0:  # sample during training --> already above so delete (later). 
            p = y_hat_n.squeeze(1)  # (batch, num_labels)

            discrete_samples = torch.transpose(
                torch.distributions.Bernoulli(probs=p).sample(
                    [self.num_samples]  # (num_samples, batch, num_labels)
                ),
                0,
                1,
            )  # (batch, num_samples, num_labels)
            if self.keep_probs:
                samples = torch.cat(
                    (discrete_samples, y_hat_n), dim=1
                )  # (batch, num_samples+1, num_labels)
            else:
                samples = discrete_samples
        else: # if self.num_sample <= 0 then return None.
            samples = None 
        
        return samples

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if labels is None or (not self.training):
            y_inf: torch.Tensor = self.inference_nn(x, buffer).unsqueeze(
                1
            )  # (batch_size, 1, ...)

            return y_inf, None
        else:
            # switch on gradients on the parameters of inference network using context manager
            with self.only_inference_nn_grad_on():
                labels = labels.unsqueeze(1)
                # loss_fn = self.get_loss_fn(
                #     x, labels
                # )  #: Loss function will expect labels in form (batch, num_samples or 1, ...)
                ## This part is removed. 
                loss_values: List[float] = []
                step_number = 0
                loss_value: Union[torch.Tensor, float] = float("inf")

                while not self.stopping_criteria(
                    step_number, float(loss_value)
                ):
                    self.optimizer.zero_grad(set_to_none=True)
                    y_inf: torch.Tensor = self.inference_nn(
                        x, buffer
                    ).unsqueeze(
                        1
                    )  # (batch_size, 1, ...) unormalized
                    # inference_nn is TaskNN so it will output tensor of shape (batch, ...)
                    # hence the unsqueeze

                    if self.cost_augmented_layer is not None:
                        y_cost_aug = self.cost_augmented_layer(
                            torch.cat(
                                (
                                    y_inf.squeeze(1),
                                    labels.to(dtype=y_inf.dtype).squeeze(1),
                                ),
                                dim=-1,
                            ),
                            buffer,
                        ).unsqueeze(
                            1
                        )  # (batch_size,1, ...)
                    else:
                        y_cost_aug = None
                        
                    samples = self.draw_samples(y_inf)
                    loss_value = self.update( # made self.update to be the same as Loss class forward()
                        x, labels, samples, y_inf, y_cost_aug, buffer 
                    )
                    loss_values.append(float(loss_value))

                    step_number += 1
            # once out of Sampler, y_inf and y_cost_aug should not get gradients
            return (
                y_inf.detach().clone(),
                y_cost_aug.detach().clone()
                if y_cost_aug is not None
                else None,
            )    
    
    def update(
        self, 
        x: Any, 
        labels: torch.Tensor, 
        samples: torch.Tensor,
        y_hat: torch.Tensor, 
        y_cost_aug: torch.Tensor, 
        buffer: Dict,
    ) -> torch.Tensor:
        """
        Returns:
            loss: loss value at the previous point (unreduced)
        """
        total_loss = self.loss_fn(
                x,
                labels,
                y_hat,
                y_cost_aug,
                buffer,
            ) # (batch, 1, num_labels)
        if self.num_samples>0:
            loss_samples = self.loss_fn_sample(
                    x,
                    samples,
                    y_hat.expand_as(samples),
                    None, # y_cost_aug shouldn't be calculated.
                    buffer,
                ) # (batch, num_samples, num_labels)
            grad_samples = self.get_sample_grads(
                    x,
                    samples,
                    buffer,
            ) # (batch, num_samples, num_labels) grab gradients w.r.t.samples from score loss.

            loss_samples = grad_samples*loss_samples
            total_loss = total_loss + torch.sum(self.sample_loss_weight * loss_samples) # shouldn't it be mean?
        total_loss.backward()  # type:ignore
        # y_inf.expand_as(pseudo_labels)
        self.optimizer.step()

        return total_loss

    def get_sample_grads(
        self, 
        x: Any, 
        samples: torch.Tensor,
        buffer: Dict,
    ) -> torch.Tensor:
        """
        Returns:
            loss: loss value at the previous point (unreduced)
        """
        
        samples.requires_grad = True
        assert (
            samples.requires_grad
        ), "Samples to grab gradient should have requires_grad=True"
        # if I get error here, I should fix it later.

        with torch.enable_grad():
            samples.grad = None # --> maybe not needed
            loss_for_grad = self.loss_fn_for_grad(
                x,
                None, # labels not required to get score-NN score.
                samples,
                None, # y_cost_aug shouldn't be used for gradient grabbing.
                buffer,
            )
            loss = torch.mean(loss_for_grad) #ToDo: change this mean to torch.sum(), and chagne sum() on the update funciton. 
            grad_samples = torch.autograd.grad(outputs=loss, inputs=samples, only_inputs=True   )
        return grad_samples[0].clone().detach() # grad returns tuple. list of length 1.

    def get_metrics(self, reset: bool = False):
        metrics = self._metrics
        metrics['total_' + self.name + '_loss'] = float(
            self._total_loss / self._num_batches) if self._num_batches > 0 else 0.0
        if reset:
            self._metrics = {}
            self._total_loss = 0.0
            self._num_batches = 0
            metrics.pop(self.name + '_loss', None)
        else:
            loss_metrics = self.loss_fn.get_metrics(reset=True)
            metrics.update(loss_metrics)

        return metrics


# print("self.sample_loss_weight {}".format(self.sample_loss_weight))
# print("self.num_samples {}".format(self.num_samples))
# print(samples.size())
# print(grad_samples)
# print(len(grad_samples))
# print(grad_samples[0].size())
# print("===========len(grad_samples): {}===========".format(len(grad_samples)))
