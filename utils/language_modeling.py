'''
This file is from transformers.data.datasets.data_collator
'''

from filelock import FileLock
import glob
import os
import pickle
import random
import re
import time
from typing import Dict, List
import warnings

import numpy as np
import datasets
import nltk
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging



logging.enable_explicit_format()
# logger = logging.get_logger(__name__)
# logger.setLevel(logging.get_verbosity())
logger = logging.get_logger()


DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)

# common class between NSP and LBL
class AbstractDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        cached_features_file: str,
        block_size: int,
        overwrite_cache:bool,
        short_seq_probability:float = 0.1,
    ):
        self.short_seq_probability = short_seq_probability
        directory, filename = os.path.split(file_path)
        self.tokenizer = tokenizer

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(np.array(tokens, dtype='int32'))

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(self, document, doc_index, block_size)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )
    # def create_examples_from_document(self):
    #     pass
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]                

class TextDatasetForNextSentencePrediction(AbstractDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache:bool,
        short_seq_probability:float = 0.1,
        nsp_probability:float = 0.5,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_nsp_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )
        # Overwride TextDatasetForNextSentencePrediction.create_examples_from_document
        self.create_examples_from_document = create_examples_from_document_for_nsp

        super().__init__(
            tokenizer, file_path, cached_features_file, block_size, 
            overwrite_cache, short_seq_probability
        )
        

# Overwride TextDatasetForNextSentencePrediction.create_examples_from_document
def create_examples_from_document_for_nsp(self, document: List[np.ndarray], doc_index: int, block_size: int):
    """Creates examples for a single document."""

    max_num_tokens = block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

    # We *usually* want to fill up the entire sequence since we are padding
    # to `block_size` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pretraining and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `block_size` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < self.short_seq_probability:
        target_seq_length = random.randint(2, max_num_tokens)

    current_chunk = []  # a buffer stored current working segments
    current_length = 0
    i = 0

    while i < len(document):
        segment = document[i].tolist()
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(0, len(self.documents) - 1)
                        if random_document_index != doc_index:
                            '''
                            THIS IS CHANGED POINT
                            Confirm random_document having one more element(s)
                            '''
                            # break
                            random_document = self.documents[random_document_index]
                            if len(random_document) > 0:
                                break

                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j].tolist())
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                    """Truncates a pair of sequences to a maximum sequence length."""
                    while True:
                        total_length = len(tokens_a) + len(tokens_b)
                        if total_length <= max_num_tokens:
                            break
                        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                        assert len(trunc_tokens) >= 1
                        # We want to sometimes truncate from the front and sometimes from the
                        # back to add more randomness and avoid biases.
                        if random.random() < 0.5:
                            del trunc_tokens[0]
                        else:
                            trunc_tokens.pop()

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # add special tokens
                input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                # add token type ids, 0 for sentence a, 1 for sentence b
                token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                # 後でcastする
                example = {
                    "input_ids": torch.tensor(input_ids, dtype=torch.int),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.uint8),
                    "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.uint8),
                }

                self.examples.append(example)

            current_chunk = []
            current_length = 0

        i += 1


class LineByLineTextDataset(AbstractDataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int,
        overwrite_cache:bool, short_seq_probability:float = 0.05,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path) or os.path.isdir(file_path), f"Input file or directory path {file_path} not found"

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_lbl_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )
        self.create_examples_from_document = create_examples_from_document_for_lbl
        super().__init__(
            tokenizer, file_path, cached_features_file, block_size, 
            overwrite_cache, short_seq_probability
        )

def create_examples_from_document_for_lbl(self, document: List[np.ndarray], doc_index: int, block_size: int):
    """Creates examples for a single document."""

    max_num_tokens = block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

    # We *usually* want to fill up the entire sequence since we are padding
    # to `block_size` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pretraining and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `block_size` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < self.short_seq_probability:
        target_seq_length = random.randint(5, max_num_tokens)

    current_chunk = []  # a buffer stored current working segments
    current_length = 0
    i = 0

    while i < len(document):
        segment = document[i].tolist()
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                    """Truncates a pair of sequences to a maximum sequence length."""
                    while True:
                        total_length = len(tokens_a) + len(tokens_b)
                        if total_length <= max_num_tokens:
                            break
                        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                        assert len(trunc_tokens) >= 1
                        # We want to sometimes truncate from the front and sometimes from the
                        # back to add more randomness and avoid biases.
                        if random.random() < 0.5:
                            del trunc_tokens[0]
                        else:
                            trunc_tokens.pop()

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 0

                # add special tokens
                input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                # add token type ids, 0 for sentence a, 1 for sentence b
                token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                # 後でcastする
                example = {
                    "input_ids": torch.tensor(input_ids, dtype=torch.int),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.uint8),
                }

                self.examples.append(example)

            current_chunk = []
            current_length = 0

        i += 1

# common class between NSP and LBL
class HFAbstractDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        corpus_name: str,
        directory: str,
        cached_features_file: str,
        block_size: int,
        overwrite_cache:bool,
        short_seq_probability:float = 0.1,
    ):
        self.short_seq_probability = short_seq_probability
        self.tokenizer = tokenizer
        nltk.download('punkt')

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                if corpus_name == "wikipedia":
                    logger.info("wikpedia version: 20200501.en is used")
                    loaded_dataset = datasets.load_dataset(corpus_name, '20200501.en', cache_dir=directory, split='train[50%:52%]')['text']
                else:
                    loaded_dataset = datasets.load_dataset(corpus_name, cache_dir=directory, split='train[50%:52%]')['text']

                self.documents = [[]]
                for d in loaded_dataset:
                    # if corpus_name == 'openwebtext':
                    #     d = d["text"]
                    for paragraph in d.split("\n"):
                        if len(paragraph) < 80:
                            continue
                        for sentence in nltk.sent_tokenize(paragraph):
                            # () is remainder after link in it filtered out
                            sentence = sentence.replace("()","")
                            if re.sub(r"\s", "", sentence) == "":
                                continue
                            tokens = tokenizer.tokenize(sentence)
                            tokens = tokenizer.convert_tokens_to_ids(tokens)
                            if tokens:
                                self.documents[-1].append(np.array(tokens, dtype='int32'))
                        self.documents.append([])
                self.documents.pop(-1)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(self, document, doc_index, block_size)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]  

class HFTextDatasetForNextSentencePrediction(HFAbstractDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        corpus_name: str,
        directory: str,
        block_size: int,
        overwrite_cache:bool,
        short_seq_probability:float = 0.1,
        nsp_probability:float = 0.5,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )

        self.nsp_probability = nsp_probability

        cached_features_file = os.path.join(
            directory,
            f"cached_nsp_{tokenizer.__class__.__name__}_{block_size}_{corpus_name}",
        )
        # Overwride TextDatasetForNextSentencePrediction.create_examples_from_document
        self.create_examples_from_document = create_examples_from_document_for_nsp
        super().__init__(
            tokenizer, corpus_name, directory, cached_features_file, 
            block_size, overwrite_cache, short_seq_probability
        )

class HFLineByLineTextDataset(HFAbstractDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        corpus_name: str,
        directory: str,
        block_size: int,
        overwrite_cache:bool,
        short_seq_probability:float = 0.1,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.nsp_probability = nsp_probability

        cached_features_file = os.path.join(
            directory,
            f"cached_lbl_{tokenizer.__class__.__name__}_{block_size}_{corpus_name}",
        )
        # Overwride TextDatasetForNextSentencePrediction.create_examples_from_document
        self.create_examples_from_document = create_examples_from_document_for_lbl
        super().__init__(
            tokenizer, corpus_name, directory, cached_features_file, 
            block_size, overwrite_cache, short_seq_probability
        )
