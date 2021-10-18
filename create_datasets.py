import argparse
import random
from typing import Optional, Union
import os


import datasets
import torch
# from torch.utils.data.dataset import Dataset
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, normalizers
from tokenizers.processors import BertProcessing
import transformers
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import (
    AutoTokenizer
    BertJapaneseTokenizer
)
from transformers.utils import logging

transformers.logging.set_verbosity_info()
transformers.logging.enable_explicit_format()

import utils

logging.enable_explicit_format()
logger = logging.get_logger()


def load_tokenizer(
    tokenizer_name_or_path:str,
    tokenizer_type:str,
    mecab_dic_type:str
) -> transformers.tokenization_utils_base.PreTrainedTokenizerBase:

    if  and os.path.isfile(tokenizer_name_or_path+ "vocab.txt"):
        tokenizer_name_or_path = os.path.join(tokenizer_name_or_path, "vocab.txt")
    if os.path.isdir(tokenizer_name_or_path) or os.path.isfile(tokenizer_name_or_path):
        # load from local file
        if tokenizer_type=="sentencepiece":
            if os.path.isdir(tokenizer_name_or_path):
                tokenizer_dir = tokenizer_name_or_path
            else:
                tokenizer_dir = os.path.dirname(tokenizer_name_or_path)
            tokenizer = SentencePieceBPETokenizer(
                os.path.join(tokenizer_dir, "vocab.json"),
                os.path.join(tokenizer_dir, "merges.txt"),
                unk_token="[UNK]",
                add_prefix_space=False, # 文頭に自動でスペースを追加しない
            )
            # 改行がinput_fileにあるとtokenも改行がついてくるのでstrip
            # cf. https://github.com/huggingface/tokenizers/issues/231
            tokenizer.normalizer = normalizers.Sequence([
                normalizers.Strip(),
                normalizers.NFKC()
            ])
            # post process tokenizer
            tokenizer._tokenizer.post_processor = BertProcessing(
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
            )
            tokenizer.enable_truncation(max_length = MAX_LENGTH)
            # convert to transformers style
            tokenizer = transformers.PreTrainedTokenizerFast(
                tokenizer_object = tokenizer,
                model_max_length = MAX_LENGTH,
                unk_token = "[UNK]",
                sep_token = "[SEP]",
                pad_token = "[PAD]",
                cls_token = "[CLS]",
                mask_token = "[MASK]",
            )
        elif tokenizer_type=="wordpiece":
            # currently supports only japanese
            if os.path.isdir(tokenizer_name_or_path):
                tokenizer_name_or_path = os.path.join(tokenizer_name_or_path, "vocab.txt")
            tokenizer = transformers.BertJapaneseTokenizer(
                tokenizer_name_or_path,
                do_lower_case = False,
                word_tokenizer_type = "mecab",
                subword_tokenizer_type = "wordpiece",
                tokenize_chinese_chars = False,
                mecab_kwargs = {"mecab_dic": mecab_dic_type},
                model_max_length = MAX_LENGTH
            )
        else:
            raise ValueError(f"Invalid tokenizer_type {tokenizer_type}.")
    else:
        # load from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    return tokenizer

def make_dataset(
    input_corpus:str,
    input_file:str,
    dataset_type:str,
    dataset_dir:str,
    cache_dir:str,
    over_write:bool
) -> Union[torch.utils.data.dataset.Dataset, datasets.dataset_dict.DatasetDict, datasets.arrow_dataset.Dataset, datasets.dataset_dict.IterableDatasetDict, datasets.iterable_dataset.IterableDataset]:

    inter_dataset_path = os.path.join(dataset_dir, "inter_" + input_corpus)
    processed_dataset_path = os.path.join(dataset_dir, f"processed_{dataset_type}_{MAX_LENGTH}_{input_corpus}")
    
    if os.path.isdir(inter_dataset_path) and not over_write:
        # load sentences
        inter_dataset = datasets.load_from_disk(inter_dataset_path)
        logger.info(f"dataset loaded from {inter_dataset_path}")
    else:
        # make sentences
        documents = [[]]
        if input_corpus in ["wiki-en", "openwebtext"]:
            if input_corpus == "wiki-en":
                dataset = datasets.load_dataset("wikipedia", "20200501.en", cache_dir=cache_dir, split="train")["text"]
            elif input_corpus == "openwebtext":
                dataset = datasets.load_dataset("openwebtext", cache_dir=cache_dir, split="train")["text"]
            else:
                raise ValueError(f"Invalid input_corpus, got {input_corpus}")
            import nltk
            for d in dataset:
                for paragraph in d.split("\n"):
                    if len(paragraph) < 80:
                        continue
                    for sentence in nltk.sent_tokenize(paragraph):
                        # () is remainder after link in it filtered out
                        sentence = sentence.replace("()","")
                        if re.sub(r"\s", "", sentence) == "":
                            continue
                        documents[-1].append(sentence)
                    documents.append([])
            if documents[-1] == []:
                documents.pop(-1)
        else:
            with open(input_file, encoding="utf-8") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    # Empty lines are used as document delimiters
                    if not line and len(documents[-1]) != 0:
                        documents.append([])
                    documents[-1].append(line)
            if documents[-1] == []:
                documents.pop(-1)
        # save intermiediate
        inter_dataset = datasets.Dataset.from_dict({"sentence": documents})
        del documents
        inter_dataset.save_to_disk(inter_dataset_path)
        logger.info(f"dataset saved in {inter_dataset_path}")
    
    # tokenize
    tokenized_dataset = inter_dataset.map(_sentence_to_ids, num_proc=os.cpu_count())
    tokenized_dataset.remove_columns("sentence")

    # create_examples_from_document
    if dataset_type == "linebyline":
        processed_dataset = tokenized_dataset.map(_create_examples_from_document_for_linebyline, num_proc=os.cpu_count(), batched=True, batch_size=1, remove_columns=tokenized_dataset.columns_names)
    elif dataset_type == "nsp":
        global REF_DATASET
        REF_DATASET = tokenized_dataset.copy()
        processed_dataset = tokenized_dataset.map(lambda example, idx: _create_examples_from_document_for_nsp(), batched=True, batch_size=1, remove_columns=tokenized_dataset.columns_names)
    else:
        raise ValueError(f"Invalid dataset_type, got {dataset_type}")
    # save processed data
    processed_dataset.save_to_disk(processed_dataset_path)


def _sentence_to_ids(example):
    tokens = [TOKENIZER.tokenize(line) for line in example["sentence"]]
    tokens = [TOKENIZER.convert_tokens_to_ids(tk) for tk in tokens]
    return {"tokens": tokens}

def _create_examples_from_document_for_linebyline(document):
    """Creates examples for a single document."""
    block_size = MAX_LENGTH
    max_num_tokens = block_size - TOKENIZER.num_special_tokens_to_add(pair=False)

    # We *usually* want to fill up the entire sequence since we are padding
    # to `block_size` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pretraining and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `block_size` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < SHORT_SEQ_PROBABILITY:
        target_seq_length = random.randint(5, max_num_tokens)

    current_chunk = []  # a buffer stored current working segments
    current_length = 0
    i = 0
    input_ids, token_type_ids = [], []

    while i < len(document):
        segment = document[i]["tokens"].tolist()
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                tokens = []
                for tk in current_chunk:
                    tokens.extend(tk)

                """Truncates a pair of sequences to a maximum sequence length."""
                while True:
                    total_length = len(tokens)
                    if total_length <= max_num_tokens:
                        break
                    # We want to sometimes truncate from the front and sometimes from the
                    # back to add more randomness and avoid biases.
                    if random.random() < 0.5:
                        del tokens[0]
                    else:
                        tokens.pop()

                assert len(tokens) >= 1

                # add special tokens
                input_ids.append(TOKENIZER.build_inputs_with_special_tokens(tokens))
                # add token type ids, 0 for sentence a, 1 for sentence b
                token_type_ids.append(TOKENIZER.create_token_type_ids_from_sequences(tokens))

            current_chunk = []
            current_length = 0

        i += 1
    return {"input_ids": input_ids, "token_type_ids": token_type_ids}


def _create_examples_from_document_for_nsp(document, doc_index)
    # Overwride TextDatasetForNextSentencePrediction.create_examples_from_document
    """Creates examples for a single document."""
    block_size = MAX_LENGTH
    max_num_tokens = block_size - TOKENIZER.num_special_tokens_to_add(pair=True)

    # We *usually* want to fill up the entire sequence since we are padding
    # to `block_size` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pretraining and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `block_size` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < SHORT_SEQ_PROBABILITY:
        target_seq_length = random.randint(2, max_num_tokens)

    current_chunk = []  # a buffer stored current working segments
    current_length = 0
    i = 0
    input_ids, token_type_ids, next_sentence_label = [], [], []

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

                if len(current_chunk) == 1 or random.random() < NSP_PROBABILITY:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(0, len(REF_DATASET) - 1)
                        if random_document_index != doc_index:
                            '''
                            THIS IS CHANGED POINT
                            Confirm random_document having one more element(s)
                            '''
                            # break
                            random_document = REF_DATASET[random_document_index]["tokens"]
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
                input_ids.append(TOKENIZER.build_inputs_with_special_tokens(tokens_a, tokens_b))
                # add token type ids, 0 for sentence a, 1 for sentence b
                token_type_ids.append(TOKENIZER.create_token_type_ids_from_sequences(tokens_a, tokens_b))
                next_sentence_label.append(int(is_random_next))

            current_chunk = []
            current_length = 0

        i += 1
    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "next_sentence_label": next_sentence_label}

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="uploaded name in HuggingFace Hub or path of vocab.txt")
    parser.add_argument("--input_corpus", type=str, required=True, choices=["wiki-ja", "wikifin-ja", "wiki-en", "openwebtext"])
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=["linebyline", "nsp"])
    parser.add_argument("--dataset_dir", type=str, required=True, help="directory which saves each dataset")
    
    # optional
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--tokenizer_type", type=str, default="", choices=["", "sentencepiece", "wordpiece"])
    parser.add_argument("--mecab_dic_type", type=str, default="", choices=["", "unidic_lite", "unidic", "ipadic"])
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets/")
    parser.add_argument("--over_write", action="store_true")

    assert args.input_corpus not in ["wiki-en", "openwebtext"] and args.input_file != "", "input_corpus must be specified with english corpus"

    # global variables
    SHORT_SEQ_PROBABILITY = 0.1
    NSP_PROBABILITY = 0.5
    MAX_LENGTH = args.max_length

    TOKENIZER = load_tokenizer(
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        tokenizer_type=args.tokenizer_type,  
        mecab_dic_type=args.mecab_dic_type,
    )

    dataset = make_dataset(
        input_corpus=args.input_corpus,
        input_file=args.input_file,
        dataset_type=args.dataset_type,
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
        over_write=args.over_write
    )
