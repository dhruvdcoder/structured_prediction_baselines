import logging
from typing import Any, Optional, Dict, List, Tuple
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import RegularizerApplicator, InitializerApplicator, util
from allennlp.nn.util import (
    viterbi_decode,
    get_lengths_from_binary_sequence_mask,
)
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import allowed_transitions

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.score_nn import ScoreNN
from .base import ScoreBasedLearningModel
from allennlp_models.structured_prediction.metrics.srl_eval_scorer import (
    SrlEvalScorer,
    DEFAULT_SRL_EVAL_PATH,
)

logger = logging.getLogger(__name__)


class SequenceTaggingModel(ScoreBasedLearningModel):
    """Abstract base class for tagging tasks like NER and SRL"""

    def __init__(
        self,
        vocab: Vocabulary,
        sampler: Sampler,
        loss_fn: Loss,
        label_encoding: str,
        label_namespace: str = "labels",
        **kwargs: Any,
    ):
        super().__init__(vocab, sampler, loss_fn, **kwargs)
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.label_namespace = label_namespace

        if not label_encoding:
            raise ConfigurationError("label_encoding was not specified.")
        self.label_encoding = label_encoding
        self.instantiate_metrics()

    def instantiate_metrics(self) -> None:
        """
        Set task appropriate metric instances on self.
        """
        raise NotImplementedError

    @overrides
    def convert_to_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Converts sequence labels from indices to one hot by adding a trailing dimension"""
        labels = F.one_hot(labels, num_classes=self.num_tags)

        return labels

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unsqueeze to get the samples dimension"""

        return labels.unsqueeze(1)

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        """Squeeze the samples dimension"""

        return y.squeeze(1)

    def constrained_decode(
        self,
        class_probabilities: torch.Tensor,  # y_hat (batch, seq_len, num_labels)
        mask: torch.BoolTensor,  # (batch, seq_len)
        wordpiece_offsets: Optional[List[List[int]]] = None,
    ) -> Tuple[
        List[List[str]], List[List[int]], List[List[str]], List[List[int]]
    ]:
        """
        Uses the logic in https://github.com/allenai/allennlp-models/blob/main/allennlp_models/structured_prediction/models/srl_bert.py#L203


        Args:
            class_probabilities: Probability for each class for each token.
            mask: Mask indicating sequence length.
            wordpiece_offsets: Indices of the first token of a word-piece sequence representing a word.
                This is useful only for BERT embedder that produces representation for word pieces instead of words. Set to none if the tokenization keeps full words.


        Returns:
            word_tags: Tags for complete words obtained by taking the tag of first piece in the wordpiece representation.
            word_ids: ids corresponding to `word_tags`
            wordpiece_tags: Tags for each wordpiece. If wordpiece_offsets is None, then this is the same as word_tags.
            wordpiece_ids: ids corresponding to `wordpiece_ids`

        """
        all_predictions = class_probabilities
        sequence_lengths = get_lengths_from_binary_sequence_mask(
            mask
        ).data.tolist()  # list with seq len for each instance in the batch.

        if wordpiece_offsets is None:
            wordpiece_offsets = [
                list(range(length)) for length in sequence_lengths
            ]

        # seperate out the instances in the batch into a list

        if all_predictions.dim() == 3:
            predictions_list = [
                all_predictions[i].detach().cpu()
                for i in range(all_predictions.size(0))
            ]
        else:  # There is no batch dimension.
            predictions_list = [all_predictions]

        wordpiece_tags: List[List[str]] = []
        word_tags: List[List[str]] = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        end_transitions = self.get_end_transitions()
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        word_ids: List[List[int]] = []
        wordpiece_ids: List[List[int]] = []

        for predictions, length, offsets in zip(
            predictions_list,
            sequence_lengths,
            wordpiece_offsets,
        ):
            max_likelihood_sequence, _ = viterbi_decode(
                predictions[:length],
                transition_matrix,
                allowed_start_transitions=start_transitions,
                allowed_end_transitions=end_transitions,
            )  # List with ids of tags
            # Convert the list of ids to tensor

            tags: List[str] = [
                self.vocab.get_token_from_index(x, namespace="labels")
                for x in max_likelihood_sequence
            ]

            wordpiece_ids.append(max_likelihood_sequence)
            word_ids.append([max_likelihood_sequence[i] for i in offsets])
            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])

        return word_tags, word_ids, wordpiece_tags, wordpiece_ids

    def calculate_metrics_for_task(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        mask: torch.BoolTensor,
        results: Dict,
        word_tags: List[List[str]],
        word_ids: List[List[int]],
        buffer: Dict,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    @overrides
    def calculate_metrics(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:
        # y_hat: (batch, seq_len, num_labels)
        # labels: (batch, seq_len, num_labels) ie one-hot
        # mask: (batch, seq_len)
        mask = buffer.get("mask")
        assert mask is not None
        metadata = buffer.get("meta")
        assert metadata is not None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        (
            word_tags,
            word_ids,
            wordpiece_tags,
            wordpiece_ids,
        ) = self.constrained_decode(
            y_hat, mask, buffer.get("wordpiece_offsets")
        )
        # call specialized computation implemented by child class
        self.calculate_metrics_for_task(  # type: ignore
            x,
            labels,
            y_hat,
            mask,
            results,
            word_tags,
            word_ids,
            buffer,
            **kwargs,
        )

    def get_viterbi_pairwise_potentials(self) -> torch.Tensor:
        """
        Generate a matrix of pairwise transition potentials for the BIOUL or BIO labels.

        Returns:
            transition_matrix : `torch.Tensor`
                A `(num_labels, num_labels)` matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        transition_matrix = torch.ones([self.num_tags, self.num_tags])
        transition_matrix *= float("-inf")
        transitions = allowed_transitions(
            self.label_encoding, all_labels
        )  # List[Tuple[int, int]]` The allowed transitions (from_label_id, to_label_id)

        for from_label, to_label in transitions:
            if from_label < self.num_tags and to_label < self.num_tags:
                transition_matrix[from_label][to_label] = 0

        return transition_matrix

    def get_start_transitions(self) -> torch.Tensor:
        """
        Allowed start transitions for BIO or BIOUL.

        1. In the BIOUL sequence, we cannot start the sequence with an I-XXX or L-XXX tag.
        2. In the BIO sequence, we cannot start the sequence with an I-XXX tag.

        This transition sequence is passed to viterbi_decode to specify this constraint.

        Returns:
            start_transitions : `torch.Tensor` 1-D tensor with the pairwise potentials
                between a START token and
                the first token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I" or label[0] == "L":
                start_transitions[i] = float("-inf")

        return start_transitions

    def get_end_transitions(self) -> torch.Tensor:
        """
        In the BIOUL or BIO sequence, we cannot end the sequence with an I-XXX or B-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.

        Returns:

            end_transitions : 1-D `torch.Tensor` of pairwise potentials between a END token and
                the last token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        end_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I" or label[0] == "B":
                end_transitions[i] = float("-inf")

        return end_transitions


@Model.register(
    "seal-ner-multi-task",
    constructor="from_partial_objects_with_shared_tasknn"
)
@Model.register(
    "seal-ner",
    constructor="from_partial_objects_with_inference_module_as_sampler",
)
@Model.register(
    "ner-seperate-inference-and-training-network",
    constructor="from_partial_objects",
)
class NERModel(SequenceTaggingModel):
    @overrides
    def instantiate_metrics(self) -> None:
        self._f1_metric = SpanBasedF1Measure(
            self.vocab,
            tag_namespace=self.label_namespace,  # type: ignore
            label_encoding=self.label_encoding,  # type: ignore
        )
        self._accuracy = CategoricalAccuracy()

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("tokens")
        _forward_args["buffer"]["mask"] = util.get_text_field_mask(
            _forward_args["x"]
        )
        _forward_args["labels"] = kwargs.pop("tags")
        _forward_args["meta"] = kwargs.pop("metadata")
        _forward_args["buffer"]["meta"] = _forward_args["meta"]

        return {**_forward_args, **kwargs}

    @overrides
    def calculate_metrics_for_task(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        mask: torch.BoolTensor,
        results: Dict,
        word_tags: List[List[str]],
        word_ids: List[List[int]],
        buffer: Dict,
        **kwargs: Any,
    ) -> None:
        true_sequence_lengths = get_lengths_from_binary_sequence_mask(
            mask
        ).data.tolist()

        masked_sequence_length = mask.shape[-1]
        padded_predicted_ids: List[torch.Tensor] = []

        for predicted_ids, length in zip(word_ids, true_sequence_lengths):
            assert len(predicted_ids) == length
            padded_predicted_ids.append(
                self.convert_to_one_hot(
                    F.pad(
                        torch.tensor(
                            predicted_ids,
                            dtype=torch.int64,
                            device=labels.device,
                        ),
                        (0, masked_sequence_length - length),
                        "constant",
                        0,
                    ).reshape(1, -1)
                )  # reshape to (1, seq_len) prepare for cat on the batch dim
            )  # DP: This might be a little slow. JY,ZY,PG, feel free to make it more efficient.
        prediction_tensor = torch.cat(padded_predicted_ids, dim=0)
        labels_indices = torch.argmax(
            labels, dim=-1
        )  # because the labels here will be one-hot but metric requires indices.
        self._f1_metric(prediction_tensor, labels_indices, mask)
        self._accuracy(prediction_tensor, labels_indices, mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_dict = self._f1_metric.get_metric(reset=reset)
        metrics = {x: y for x, y in f1_dict.items() if "overall" in x}
        metrics["accuracy"] = self._accuracy.get_metric(reset=reset)

        return metrics


@Model.register(
    "seal-srl-multi-task",
    constructor="from_partial_objects_with_shared_tasknn"
)
@Model.register(
    "seal-srl",
    constructor="from_partial_objects_with_inference_module_as_sampler",
)
@Model.register(
    "srl-seperate-inference-and-training-network",
    constructor="from_partial_objects",
)
class SRLModel(SequenceTaggingModel):
    def __init__(
        self,
        srl_eval_path: Optional[str] = DEFAULT_SRL_EVAL_PATH,
        using_bert_encoder: bool = True,
        decode_on_wordpieces: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            srl_eval_path: Path to the perl evaluation file used by allennlp

        """
        self.srl_eval_path = srl_eval_path
        self.using_bert_encoder = using_bert_encoder
        self.decode_on_wordpieces = decode_on_wordpieces
        super().__init__(**kwargs)

        if self.decode_on_wordpieces:
            assert self.using_bert_encoder

    @overrides
    def instantiate_metrics(self) -> None:
        if self.srl_eval_path is not None:
            # For the span based evaluation, we don't want to consider labels
            # for verb, because the verb index is provided to the model.
            self.span_metric = SrlEvalScorer(
                self.srl_eval_path, ignore_classes=["V"]
            )
        else:
            logger.warning(
                "There will be no evaluation because srl_eval_path is not provided."
            )
            self.span_metric = None

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args: Dict[str, Any] = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        tokens = kwargs.pop("tokens")

        # !!!!!!!!!!     This is a hack and is brittle !!!!!!!!!!!!!!!!!! #
        # This will not work for any other kind of embedder other than bert
        # Ideally we should specialize the PretrainedTransformerTokenEmbedder
        # and PretraintedTransformerMistmatched to take extra input
        # The current hack achieve what the following code does:
        # https://github.com/allenai/allennlp-models/blob/v2.5.0/allennlp_models/structured_prediction/models/srl_bert.py#L143

        if self.using_bert_encoder:
            tokens["tokens"]["type_ids"] = kwargs.pop("verb_indicator")
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

        _forward_args["x"] = tokens
        _forward_args["buffer"]["mask"] = util.get_text_field_mask(
            _forward_args["x"]
        )
        _forward_args["labels"] = kwargs.pop("tags")

        metadata = kwargs.pop("metadata")

        if metadata is None:
            raise ValueError
        _forward_args["buffer"]["meta"] = metadata

        if self.decode_on_wordpieces:

            if len(metadata) > 0:
                assert "offsets" in metadata[0]
                _forward_args["buffer"]["wordpiece_offsets"] = [
                    x["offsets"] for x in metadata
                ]

        return {**_forward_args, **kwargs}

    @overrides
    def calculate_metrics_for_task(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        mask: torch.BoolTensor,
        results: Dict,
        word_tags: List[List[str]],
        word_ids: List[List[int]],
        buffer: Dict,
        **kwargs: Any,
    ) -> None:
        if self.span_metric is not None and not self.training:
            metadata = buffer.get("meta")
            assert metadata is not None
            from allennlp_models.structured_prediction.models.srl import (
                convert_bio_tags_to_conll_format,
            )

            batch_conll_predicted_tags = [
                convert_bio_tags_to_conll_format(tags) for tags in word_tags
            ]
            batch_bio_gold_tags = [
                example_metadata["gold_tags"] for example_metadata in metadata
            ]
            batch_conll_gold_tags = [
                convert_bio_tags_to_conll_format(tags)
                for tags in batch_bio_gold_tags
            ]
            batch_verb_indices = [
                example_metadata["verb_index"] for example_metadata in metadata
            ]
            batch_sentences = [
                example_metadata["words"] for example_metadata in metadata
            ]
            self.span_metric(
                batch_verb_indices,
                batch_sentences,
                batch_conll_predicted_tags,
                batch_conll_gold_tags,
            )

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.span_metric.get_metric(reset=reset)
        # This can be a lot of metrics, as there are 3 per class.
        # we only really care about the overall metrics, so we filter for them here.

        return {x: y for x, y in metric_dict.items() if "overall" in x}
