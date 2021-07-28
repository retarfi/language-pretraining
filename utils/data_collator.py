'''
This file is from transformers.data.data_collator
'''
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import torch

from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


def _collate_batch(examples, tokenizer:PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


# Override transformers.data.data_collator.DataCollatorForWholeWordMask
@dataclass
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        mlm:bool = True,
        mlm_probability:float = 0.15,
        rate_replaced:float = 0.8,
        rate_random:float = 0.1,
        rate_unchanged:float = 0.1
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.rate_replaced = rate_replaced
        rate_random= rate_random
        rate_unchanged = rate_unchanged
        assert self.rate_replaced + rate_random + rate_unchanged == 1

        # 逆算でreplaceされてないもののうちrandomにする割合を求める
        self.rate_random_condition = self.mlm_probability * rate_random / (1 - self.mlm_probability * self.rate_replaced)
        # 逆算でreplaceとrandomでないもののうち何の処理もされないものの割合を求める
        self.rate_left_condition = (1 - self.mlm_probability) / (1 - self.mlm_probability * (self.rate_replaced + rate_random))

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        # cast to torch.long
        for k, v in batch.items():
            batch[k] = v.to(torch.long)
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # inputs:(batch, seq_len)
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self._whole_word_mask(labels, special_tokens_mask, self.rate_replaced)
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, self.rate_random_condition)).bool() & ~indices_replaced & ~special_tokens_mask
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        indices_left = torch.bernoulli(torch.full(labels.shape, self.rate_left_condition)).bool() & ~indices_replaced & ~indices_random & ~special_tokens_mask

        labels[(~indices_replaced) & (~indices_random) & (~indices_left)] = -100  # We only compute loss on masked tokens
        return inputs, labels


    def _whole_word_mask(
        self, input_ids:torch.Tensor, special_tokens_mask:torch.Tensor,
        rate_replaced:float, max_predictions:int=512
    ) -> torch.Tensor:
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # 0.15をboolにするときにsubwordを考慮すべき
        # idea: 候補となる単語をmain-wordとして抽出→shuffle→前から0.15*0.8になるまで抽出
        # →これらをmask
        mask_indices = []
        for document_ids, special_tokens in zip(input_ids.tolist(), special_tokens_mask.tolist()):
            document_tokens = self.tokenizer.convert_ids_to_tokens(document_ids)
            cand_indexes = []
            for i, (token, is_special_token) in enumerate(zip(document_tokens, special_tokens)):
                if is_special_token:
                    continue
                if token == self.tokenizer.pad_token:
                    break
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])

            np.random.shuffle(cand_indexes)
            num_to_mask = min(max_predictions, max(1, int(round(len(document_tokens) * self.mlm_probability * self.rate_replaced))))
            total_masked = 0
            covered_indexes = set()
            for index_set in cand_indexes:
                if total_masked >= num_to_mask:
                    break
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.
                if total_masked + len(index_set) > num_to_mask:
                    continue
                covered_indexes = covered_indexes | set(index_set)
                total_masked += len(index_set)

            mask = [1 if i in covered_indexes else 0 for i in range(len(document_ids))]
            mask_indices.append(mask)
        mask_indices = torch.tensor(mask_indices).bool()
        return mask_indices


@dataclass
class DataCollatorForLanguageModelingWithElectra(DataCollatorForLanguageModeling):
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.85)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (15% of the time) we keep the masked input tokens unchanged
        return inputs, labels



