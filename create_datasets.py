import argparse
import copy
import itertools
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
from datasets import Dataset
from jptranstokenizer import JapaneseTransformerTokenizer
from transformers import BatchEncoding, PreTrainedTokenizerBase

import utils
from utils.data_collator import get_mask_datacollator
from utils.logger import make_logger_setting


# logger
logger: logging.Logger = logging.getLogger(__name__)
make_logger_setting(logger)

# global variables
NSP_PROBABILITY: int = 0.5


def make_dataset(
    input_corpus: str,
    input_file: str,
    dataset_type: str,
    mask_style: str,
    tokenizer: JapaneseTransformerTokenizer,
    max_length: int,
    do_save: bool = True,
    mlm_probability: float = 0.15,
    dataset_dir: str = "./dataset",
    cache_dir: str = "./.cache/datasets",
) -> Union[
    torch.utils.data.dataset.Dataset,
    datasets.dataset_dict.DatasetDict,
    datasets.arrow_dataset.Dataset,
    datasets.dataset_dict.IterableDatasetDict,
    datasets.iterable_dataset.IterableDataset,
]:
    # assertions
    assert (
        input_corpus in ["wiki-en", "openwebtext"] or input_file != ""
    ), "input_file must be specified with japanese corpus"
    assert (
        dataset_type != "" or mask_style != "none"
    ), "dataset_type or mask_syle must be specified (except none)"
    assert (
        mask_style.split("-")[0] != "bert"
    ), "Pre-masking for bert is not available in run_pretraining.py"
    assert mlm_probability > 0 and mlm_probability < 1
    if dataset_type != "" and mask_style != "none":
        logger.warning(
            f"mask_style {mask_style} has priority to dataset_type {dataset_type}"
        )

    # make sentences
    documents: List[List[str]] = [[]]
    if input_corpus in ["wiki-en", "openwebtext"]:
        loaded_ds: dataset.DatasetDict
        if input_corpus == "wiki-en":
            loaded_ds = datasets.load_dataset(
                "wikipedia", "20200501.en", cache_dir=cache_dir, split="train"
            )["text"]
        elif input_corpus == "openwebtext":
            loaded_ds = datasets.load_dataset(
                "openwebtext", cache_dir=cache_dir, split="train"
            )["text"]
        else:
            raise ValueError(f"Invalid input_corpus, got {input_corpus}")
        import nltk

        for d in loaded_ds:
            for paragraph in d.split("\n"):
                if len(paragraph) < 80:
                    continue
                sentence: str
                for sentence in nltk.sent_tokenize(paragraph):
                    # () is remainder after link in it filtered out
                    sentence = sentence.replace("()", "")
                    if sentence and re.sub(r"\s", "", sentence) != "":
                        documents[-1].append(sentence)
                documents.append([])
    else:
        with open(input_file, encoding="utf-8") as f:
            while True:
                line: str = f.readline()
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
    ds: Dataset = Dataset.from_dict({"sentence": documents})
    del documents

    # tokenize
    ds = ds.map(
        lambda example: _sentence_to_ids(example, tokenizer, batched=False),
        remove_columns=["sentence"],
        batched=False,
        load_from_cache_file=False,
    ).flatten_indices()
    ds = ds.filter(
        lambda example: len(example["tokens"]) > 0
        and not (len(example["tokens"]) == 1 and len(example["tokens"][0]) == 0),
        # num_proc=None,
    ).flatten_indices()
    logger.info("Tokenize finished")

    # create_examples_from_document
    if dataset_type == "" or mask_style != "none":
        # decide dataset_type from mask_style
        if mask_style.split("-")[0] == "bert":
            dataset_type = "nsp"
        else:
            dataset_type = "linebyline"
    if dataset_type == "linebyline":
        ds = ds.map(
            lambda example: _create_examples_from_document_for_linebyline(
                example, tokenizer, max_length
            ),
            num_proc=None,
            batched=True,
            batch_size=1000,
            remove_columns=["tokens"],
            load_from_cache_file=False,
        )
    elif dataset_type == "nsp":
        global REF_DATASET
        REF_DATASET = copy.copy(ds)
        ds = ds.map(
            lambda example, idx: _create_examples_from_document_for_nsp(
                example, idx, tokenizer, max_length
            ),
            num_proc=None,
            batched=True,
            batch_size=1,
            with_indices=True,
            remove_columns=["tokens"],
            load_from_cache_file=False,
        )
        del REF_DATASET
    else:
        raise ValueError(f"Invalid dataset_type, got {dataset_type}")

    # apply masking
    if mask_style != "none":
        do_whole_word_mask: bool
        if mask_style[-4:] == "-wwm":
            do_whole_word_mask = True
        else:
            do_whole_word_mask = False
        data_collator = get_mask_datacollator(
            model_name=mask_style.split("-")[0],
            do_whole_word_mask=do_whole_word_mask,
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
        )
        # example: Dict[str, Union[List[int], int]]
        # data_collator: List[Dict[str, Union[List[int], int]]] -> BatchEncoding[str, torch.tensor(size: (1) or(1, max_length))]
        # _convert_batchencoding_to_dict: BatchEncoding -> Dict[str, Union[List[int], int]]:
        ds = ds.map(
            lambda example: _convert_batchencoding_to_dict(
                batch=data_collator(
                    [(example if isinstance(example, dict) else example.data)]
                ),
                tokenizer=tokenizer,
                max_length=max_length
            )
        )

    # save processed data
    if do_save:
        processed_dataset_path: str = os.path.join(
            dataset_dir, f"{dataset_type}_{max_length}_{input_corpus}"
        )
        if mask_style != "none":
            processed_dataset_path += f"_{mask_style}"
        ds.flatten_indices().save_to_disk(processed_dataset_path)
        logger.info(f"Processed dataset saved in {processed_dataset_path}")
    return ds


def _sentence_to_ids(
    example: Dict[str, Union[Any, List]],
    tokenizer: JapaneseTransformerTokenizer,
    batched: bool,
) -> Dict[str, List[str]]:
    tokens: List[Union[List[str], str]]
    if batched:
        tokens = [
            [tokenizer.tokenize(line) for line in batch]
            for batch in example["sentence"]
        ]
        tokens = [
            [tokenizer.convert_tokens_to_ids(tk) for tk in batch if tk]
            for batch in tokens
            if batch
        ]
    else:
        if tokenizer.word_tokenizer_type == "juman":
            p = re.compile("[a-zA-Z]+")
            tokens = [
                tokenizer.tokenize(line)
                for line in example["sentence"]
                if len(line.encode("utf-8")) <= 4096
                and len("".join(p.findall(line))) / len(line) < 0.85
            ]
        else:
            tokens = [tokenizer.tokenize(line) for line in example["sentence"]]
            tokens = [tokenizer.convert_tokens_to_ids(tk) for tk in tokens if tk]
    return {"tokens": tokens}


def _create_examples_from_document_for_linebyline(
    batch: Dict[str, List[List[int]]],
    tokenizer: JapaneseTransformerTokenizer,
    max_length: int,
) -> Dict[str, List[List[int]]]:
    """Creates examples for documents."""
    block_size: int = max_length
    max_num_tokens: int = block_size - tokenizer.num_special_tokens_to_add(pair=False)

    current_chunk: List[int] = []  # a buffer stored current working segments
    current_length: int = 0
    input_ids: List[List[int]] = []
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
                    input_ids.append(
                        tokenizer.build_inputs_with_special_tokens(current_chunk)
                    )
                current_chunk = []
                current_length = 0
    else:
        current_chunk = list(itertools.chain.from_iterable(current_chunk))
        if len(current_chunk) >= max_num_tokens * 0.8:
            input_ids.append(tokenizer.build_inputs_with_special_tokens(current_chunk))
    return {"input_ids": input_ids}


def _create_examples_from_document_for_nsp(
    document: List[List[int]],
    doc_index: int,
    tokenizer: JapaneseTransformerTokenizer,
    max_length: int,
) -> Dict[str, List[Union[List[int], int]]]:
    # Overwride TextDatasetForNextSentencePrediction.create_examples_from_document
    """Creates examples for a single document."""
    block_size = max_length
    max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)

    # We *usually* want to fill up the entire sequence since we are padding
    # to `block_size` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pretraining and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `block_size` is a hard limit.
    target_seq_length: int = max_num_tokens
    short_seq_probability: float = 0.1
    if random.random() < short_seq_probability:
        target_seq_length = random.randint(2, max_num_tokens)

    current_chunk: List[List[int]] = []  # a buffer stored current working segments
    current_length: int = 0
    i: int = 0
    input_ids: List[List[int]] = []
    token_type_ids: List[List[int]] = []
    next_sentence_label: List[int] = []
    # for batched process, index must be 0
    document = document["tokens"][0]
    while i < len(document):
        segment: List[int] = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a: List[int] = list(
                    itertools.chain.from_iterable(current_chunk[:a_end])
                )
                # tokens_a = []
                # for j in range(a_end):
                #     tokens_a.extend(current_chunk[j])

                tokens_b: List[int] = []
                is_random_next: bool
                if len(current_chunk) == 1 or random.random() < NSP_PROBABILITY:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index: int = random.randint(
                            0, len(REF_DATASET) - 1
                        )
                        if random_document_index != doc_index:
                            """
                            THIS IS CHANGED POINT
                            Confirm random_document having one more element(s)
                            """
                            # break
                            random_document = REF_DATASET[random_document_index][
                                "tokens"
                            ]
                            if len(random_document) > 0:
                                break

                    random_start: int = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments: int = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                    """Truncates a pair of sequences to a maximum sequence length."""
                    while True:
                        total_length: int = len(tokens_a) + len(tokens_b)
                        if total_length <= max_num_tokens:
                            break
                        trunc_tokens: List[int] = (
                            tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                        )
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
                input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                )
                # add token type ids, 0 for sentence a, 1 for sentence b
                token_type_ids.append(
                    tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
                )
                next_sentence_label.append(int(is_random_next))

            current_chunk = []
            current_length = 0

        i += 1
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "next_sentence_label": next_sentence_label,
    }


def _convert_batchencoding_to_dict(
    batch: BatchEncoding, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Dict[str, Union[List[int], int]]:
    # pad input_ids, attention_mask (, and token_type_ids if exists)
    dct: Dict[str, Union[List[int], int]] = {k: v.tolist()[0] for k, v in batch.items()}
    dct = tokenizer.pad(dct, padding="max_length", max_length=max_length).data
    ignore_index: int = -100
    # pad labels
    dct["labels"] += [ignore_index] * max(0, max_length - len(dct["labels"]))
    return dct


if __name__ == "__main__":
    # arguments
    parser: argparse.Namespace = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # required
    parser.add_argument(
        "--input_corpus",
        type=str,
        required=True,
        help=(
            "Directory name for created dataset. "
            "Other affixes are also added to this."
        ),
    )
    parser.add_argument("--max_length", type=int, required=True)

    # optional
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="",
        choices=["linebyline", "nsp", ""],
        help=(
            "This must be specified when --mask_style is none. "
            "Overwritten when --mask_sytle is other than none"
        ),
    )
    parser.add_argument("--input_file", type=str, default="")
    lst_mask_style: List[str] = ["none"] + list(
        map(
            lambda x: "".join(x),
            itertools.product(["debertav2", "electra", "roberta"], ["", "-wwm"]),
        )
    )
    parser.add_argument(
        "--mask_style",
        type=str,
        default="none",
        choices=lst_mask_style,
        help=(
            "If none (default), no masking. "
            "If other choice, masking is applied. "
            "-wwm means applying whole-word-masking. "
            "Masking for bert is not available (only dynamic masking is avialable)"
        ),
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of target for masking",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset/",
        help="Directory which saves each dataset",
    )
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets/")
    utils.add_arguments_for_tokenizer(parser)

    args: argparse.Namespace = parser.parse_args()
    utils.assert_arguments_for_tokenizer(args)

    tokenizer: JapaneseTransformerTokenizer = utils.load_tokenizer(args)

    dataset: Union[
        torch.utils.data.dataset.Dataset,
        datasets.dataset_dict.DatasetDict,
        datasets.arrow_dataset.Dataset,
        datasets.dataset_dict.IterableDatasetDict,
        datasets.iterable_dataset.IterableDataset,
    ] = make_dataset(
        input_corpus=args.input_corpus,
        input_file=args.input_file,
        dataset_type=args.dataset_type,
        mask_style=args.mask_style,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        do_save=True,
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
    )
