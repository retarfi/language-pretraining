import argparse
import copy
import itertools
import logging
import os
import random
import re
from typing import Optional, Union

import datasets
import torch

import utils


def make_dataset(
    input_corpus:str,
    input_file:str,
    dataset_type:str,
    dataset_dir:str,
    cache_dir:str,
) -> Union[torch.utils.data.dataset.Dataset, datasets.dataset_dict.DatasetDict, datasets.arrow_dataset.Dataset, datasets.dataset_dict.IterableDatasetDict, datasets.iterable_dataset.IterableDataset]:

    processed_dataset_path = os.path.join(dataset_dir, f"{dataset_type}_{MAX_LENGTH}_{input_corpus}")
    
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
                    if sentence and re.sub(r"\s", "", sentence) != "":
                        documents[-1].append(sentence)
                documents.append([])
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
                if line and re.sub(r"\s", "", line) != "":
                    documents[-1].append(line)
    if documents[-1] == []:
        documents.pop(-1)
    # save intermiediate
    inter_dataset = datasets.Dataset.from_dict({"sentence": documents})
    del documents
    
    # tokenize
    # num_proc = 3
    tokenized_dataset = inter_dataset.map(
        lambda example: _sentence_to_ids(example, TOKENIZER, batched=True), 
        remove_columns=["sentence"],
        # num_proc=num_proc,
        batched=True,
        load_from_cache_file=False
    ).flatten_indices()
    del inter_dataset
    filtered_dataset = tokenized_dataset.filter(
        lambda example: len(example["tokens"])>0 and not (len(example["tokens"])==1 and len(example["tokens"][0])==0),
        # num_proc=None,
    ).flatten_indices()
    del tokenized_dataset
    logger.info("Tokenize finished")
    
    # create_examples_from_document
    if dataset_type == "linebyline":
        processed_dataset = filtered_dataset.map(
            lambda example: _create_examples_from_document_for_linebyline(example, TOKENIZER),
            num_proc=None,
            batched=True,
            batch_size=1000,
            remove_columns=["tokens"],
            load_from_cache_file=False
        )
    elif dataset_type == "nsp":
        global REF_DATASET
        REF_DATASET = copy.copy(filtered_dataset)
        processed_dataset = filtered_dataset.map(
            lambda example, idx: _create_examples_from_document_for_nsp(example, idx, TOKENIZER),
            num_proc=None,
            batched=True,
            batch_size=1,
            with_indices=True,
            remove_columns=["tokens"],
            load_from_cache_file=False
        )
    else:
        raise ValueError(f"Invalid dataset_type, got {dataset_type}")
    # save processed data
    del filtered_dataset
    processed_dataset.flatten_indices().save_to_disk(processed_dataset_path)
    logger.info(f"Processed dataset saved in {processed_dataset_path}")


def _sentence_to_ids(example,TOKENIZER, batched):
    if batched:
        tokens = [[TOKENIZER.tokenize(line) for line in batch] for batch in example["sentence"]]
        tokens = [[TOKENIZER.convert_tokens_to_ids(tk) for tk in batch if tk] for batch in tokens if batch]
    else:
        tokens = [TOKENIZER.tokenize(line) for line in example["sentence"]]
        tokens = [TOKENIZER.convert_tokens_to_ids(tk) for tk in tokens if tk]
    return {"tokens": tokens}


def _create_examples_from_document_for_linebyline(batch, TOKENIZER):
    """Creates examples for documents."""
    block_size = MAX_LENGTH
    max_num_tokens = block_size - TOKENIZER.num_special_tokens_to_add(pair=False)

    current_chunk = []  # a buffer stored current working segments
    current_length = 0
    input_ids = []
    for document in batch["tokens"]:
        for segment in document:
            current_chunk.append(segment)
            current_length += len(segment)
            if current_length >= max_num_tokens:
                if current_chunk:
                    current_chunk = list(itertools.chain.from_iterable(current_chunk))
                    """Truncates a pair of sequences to a maximum sequence length."""
                    while True:
                        total_length = len(current_chunk)
                        if total_length <= max_num_tokens:
                            break
                        current_chunk.pop()
                    assert len(current_chunk) >= 1
                    # add special tokens
                    input_ids.append(TOKENIZER.build_inputs_with_special_tokens(current_chunk))
                current_chunk = []
                current_length = 0
    else:
        current_chunk = list(itertools.chain.from_iterable(current_chunk))
        if len(current_chunk) >= max_num_tokens * 0.8:
            input_ids.append(TOKENIZER.build_inputs_with_special_tokens(current_chunk))
    return {"input_ids": input_ids}


def _create_examples_from_document_for_nsp(document, doc_index, TOKENIZER):
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
    short_seq_probability = 0.1
    if random.random() < short_seq_probability:
        target_seq_length = random.randint(2, max_num_tokens)

    current_chunk = []  # a buffer stored current working segments
    current_length = 0
    i = 0
    input_ids, token_type_ids, next_sentence_label = [], [], []
    # for batched process, index must be 0
    document = document["tokens"][0]
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)
                
                tokens_a = list(itertools.chain.from_iterable(current_chunk[:a_end]))
                # tokens_a = []
                # for j in range(a_end):
                #     tokens_a.extend(current_chunk[j])

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
                        tokens_b.extend(random_document[j])
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
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="uploaded name in HuggingFace Hub or directory path containing vocab.txt")
    parser.add_argument("--input_corpus", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=["linebyline", "nsp"])
    
    # optional
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/", help="directory which saves each dataset")
    parser.add_argument("--tokenizer_type", type=str, default="", choices=["", "sentencepiece", "wordpiece"])
    parser.add_argument("--mecab_dic_type", type=str, default="", choices=["", "unidic_lite", "unidic", "ipadic"])
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets/")
    
    args = parser.parse_args()
    assert args.input_corpus in ["wiki-en", "openwebtext"] or args.input_file != "", "input_file must be specified with japanese corpus"

    # global variables
    datasets.config.IN_MEMORY_MAX_SIZE = 250 * 10**9
    NSP_PROBABILITY = 0.5
    MAX_LENGTH = args.max_length

    # get root logger
    # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    TOKENIZER = utils.load_tokenizer(
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        tokenizer_type=args.tokenizer_type,  
        mecab_dic_type=args.mecab_dic_type,
    )

    dataset = make_dataset(
        input_corpus=args.input_corpus,
        input_file=args.input_file,
        dataset_type=args.dataset_type,
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir
    )
