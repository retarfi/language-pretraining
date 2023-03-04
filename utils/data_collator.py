"""
This file is from transformers.data.data_collator
"""
from dataclasses import dataclass
from typing import Any, Tuple, Optional

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
)


@dataclass
class DataCollatorForLanguageModelingWithElectra(DataCollatorForLanguageModeling):
    # This changes the ratios of [MASK], replacing, and as it is from 80%, 10%, 10%
    # to 85%, 0%, 15% for ELECTRA
    # The code below is referred from v4.22.2
    def tf_mask_tokens(
        self,
        inputs: Any,
        vocab_size,
        mask_token_id,
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import tensorflow as tf

        input_shape = tf.shape(inputs)
        # 1 for a special token, 0 for a normal token in the special tokens mask
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        masked_indices = (
            self.tf_bernoulli(input_shape, self.mlm_probability) & ~special_tokens_mask
        )
        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        labels = tf.where(masked_indices, inputs, -100)

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, 0.85) & masked_indices
        inputs = tf.where(indices_replaced, mask_token_id, inputs)

        # The rest of the time (15% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.85)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # The rest of the time (15% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def numpy_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import numpy as np

        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=np.bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(np.bool)

        probability_matrix[special_tokens_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(
            1, probability_matrix, size=probability_matrix.shape
        ).astype(np.bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            np.random.binomial(1, 0.85, size=labels.shape).astype(np.bool)
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # The rest of the time (15% of the time) we keep the masked input tokens unchanged
        return inputs, labels


@dataclass
class DataCollatorForWholeWordMaskWithElectra(DataCollatorForWholeWordMask):
    # This changes the ratios of [MASK], replacing, and as it is from 80%, 10%, 10%
    # to 85%, 0%, 15% for ELECTRA
    # The code below is referred from v4.22.2
    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.85)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # The rest of the time (15% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def tf_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import tensorflow as tf

        input_shape = tf.shape(inputs)
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = tf.identity(inputs)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        masked_indices = tf.cast(mask_labels, tf.bool)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels
        ]
        masked_indices = masked_indices & ~tf.cast(special_tokens_mask, dtype=tf.bool)
        if self.tokenizer._pad_token is not None:
            padding_mask = inputs == self.tokenizer.pad_token_id
            masked_indices = masked_indices & ~padding_mask

        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        labels = tf.where(masked_indices, inputs, -100)

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, 0.85) & masked_indices

        inputs = tf.where(indices_replaced, self.tokenizer.mask_token_id, inputs)

        # The rest of the time (15% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def numpy_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import numpy as np

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        masked_indices = mask_labels.astype(np.bool)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        masked_indices[np.array(special_tokens_mask, dtype=np.bool)] = 0
        if self.tokenizer._pad_token is not None:
            padding_mask = labels == self.tokenizer.pad_token_id
            masked_indices[padding_mask] = 0

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            np.random.binomial(1, 0.8, size=labels.shape).astype(np.bool)
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # The rest of the time (15% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def get_mask_datacollator(
    model_name: str,
    do_whole_word_mask: bool,
    tokenizer: PreTrainedTokenizerBase,
    mlm_probability: float,
) -> DataCollatorMixin:
    class_datacollator: DataCollatorMixin
    if do_whole_word_mask:
        if model_name == "electra":
            class_datacollator = DataCollatorForWholeWordMaskWithElectra
        else:
            class_datacollator = DataCollatorForWholeWordMask
    else:
        if model_name == "electra":
            class_datacollator = DataCollatorForLanguageModelingWithElectra
        else:
            class_datacollator = DataCollatorForLanguageModeling
    data_collator: DataCollatorForLanguageModeling = class_datacollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    return data_collator
