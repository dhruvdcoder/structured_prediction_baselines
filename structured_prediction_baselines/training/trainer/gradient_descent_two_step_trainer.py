import logging
import math
import time
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.optim.lr_scheduler
from allennlp.common import Tqdm
from allennlp.common import util as common_util
from allennlp.data import DataLoader
from allennlp.models import Model
from allennlp.training import GradientDescentTrainer, Trainer
from allennlp.training import util as training_util
from torch.cuda import amp

from structured_prediction_baselines.modules.stopping_criteria import StoppingCriteria, StopAfterNumberOfSteps

logger = logging.getLogger(__name__)


@Trainer.register("gradient_descent_two_step", constructor="from_partial_objects")
class GradientDescentNStepTrainer(GradientDescentTrainer):
    """
    Custom Gradient Descent Trainer that allows N update steps for each batch
    """

    def __init__(self, model: Model, optimizer: torch.optim.Optimizer,
                 data_loader: DataLoader, **kwargs):
        super().__init__(model, optimizer, data_loader, **kwargs)
        self._stopping_criteria: StoppingCriteria = StopAfterNumberOfSteps(2)
        self._eval_grad = True

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []
        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage: {common_util.format_size(memory)}")
        gpu_memory_usage = []
        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage: {common_util.format_size(memory)}")

        regularization_penalty = self.model.get_regularization_penalty()

        train_loss = 0.0
        batch_loss = 0.0
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

        # Set the model to "train" mode.
        self._pytorch_model.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
        # progress is shown
        if self._primary:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing training early! "
                        "This implies that there is an imbalance in your training "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            step_number = 0
            batch_group_outputs = []
            metrics = {}
            batch_grad_norm: Optional[float] = None
            while not self._stopping_criteria(step_number, batch_loss):
                if self._eval_grad:
                    # Zero gradients.
                    # NOTE: this is actually more efficient than calling `self.optimizer.zero_grad()`
                    # because it avoids a read op when the gradients are first updated below.
                    for param_group in self.optimizer.param_groups:
                        for p in param_group["params"]:
                            p.grad = None

                    batch_loss = 0.0
                    for batch in batch_group:
                        with amp.autocast(self._use_amp):
                            batch_outputs = self.batch_outputs(batch, for_training=True)
                            batch_group_outputs.append(batch_outputs)
                            loss = batch_outputs["loss"]
                            reg_loss = batch_outputs.get("reg_loss")
                            if torch.isnan(loss):
                                raise ValueError("nan loss encountered")
                            loss = loss / len(batch_group)

                            batch_loss += loss.item()
                            if reg_loss is not None:
                                reg_loss = reg_loss / len(batch_group)
                                batch_reg_loss = reg_loss.item()
                                train_reg_loss += batch_reg_loss  # type: ignore

                        if self._scaler is not None:
                            self._scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    train_loss += batch_loss

                    batch_grad_norm = self.rescale_gradients()

                    # This does nothing if batch_num_total is None or you are using a
                    # scheduler which doesn't update per batch.
                    if self._learning_rate_scheduler:
                        self._learning_rate_scheduler.step_batch(batch_num_total)
                    if self._momentum_scheduler:
                        self._momentum_scheduler.step_batch(batch_num_total)

                    if self._scaler is not None:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        self.optimizer.step()

                    # Update moving averages
                    if self._moving_average is not None:
                        self._moving_average.apply(batch_num_total)

                    # Update the description with the latest metrics
                    metrics = training_util.get_metrics(
                        self.model,
                        train_loss,
                        train_reg_loss,
                        batch_loss,
                        batch_reg_loss,
                        batches_this_epoch,
                        world_size=self._world_size,
                        cuda_device=self.cuda_device,
                    )
                    self._eval_grad = False
                else:
                    for batch in batch_group:
                        with amp.autocast(self._use_amp):
                            self.batch_outputs(batch, for_training=False)
                    self._eval_grad = True

                step_number += 1

            if self._primary:
                # Updating tqdm only for the primary as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description, refresh=False)

                if self._checkpointer is not None:
                    self._checkpointer.maybe_save_checkpoint(self, epoch, batches_this_epoch)

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    batch_group,
                    batch_group_outputs,
                    metrics,
                    epoch,
                    batches_this_epoch,
                    is_training=True,
                    is_primary=self._primary,
                    batch_grad_norm=batch_grad_norm,
                )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        metrics = training_util.get_metrics(
            self.model,
            train_loss,
            train_reg_loss,
            batch_loss=None,
            batch_reg_loss=None,
            num_batches=batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (1024 * 1024)
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (1024 * 1024)
        return metrics
