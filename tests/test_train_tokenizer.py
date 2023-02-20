import os
import sys

import pytest

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
import train_tokenizer

INPUT_FILE: str = "data/botchan_ja.txt"
PRETOKENIZED_FILE: str = "data/botchan_ja_pretokenized.txt"
PRETOKENIZED_DIR: str = "tmp_pretokenized/"
INTERMEDIATE_DIR: str = "tmp/"


@pytest.mark.parametrize("num_files", [1, 3])
def test_split(num_files: int) -> None:
    train_tokenizer.split(
        input_file=INPUT_FILE, num_files=num_files, intermediate_dir=INTERMEDIATE_DIR
    )


@pytest.mark.parametrize(
    "num_files, word_tokenizer, sudachi_split_mode",
    [
        (1, "none", "A"),
        (3, "none", "A"),
        (1, "spacy-luw", "A"),
        (1, "mecab", "A"),
        (3, "mecab", "A"),
        (1, "juman", "A"),
        (3, "juman", "A"),
        (1, "sudachi", "A"),
        (1, "sudachi", "B"),
        (3, "sudachi", "C"),
    ],
)
def test_pre_tokenize(
    num_files: int, word_tokenizer: str, sudachi_split_mode: str
) -> None:
    train_tokenizer.pre_tokenize(
        input_file=INPUT_FILE,
        num_files=num_files,
        pretokenized_prefix="_pretokenized",
        intermediate_dir=INTERMEDIATE_DIR,
        word_tokenizer=word_tokenizer,
        mecab_dic_type="ipadic",
        mecab_option="",
        sudachi_split_mode=sudachi_split_mode,
        sudachi_config_path=None,
        sudachi_resource_dir=None,
        sudachi_dict_type="core",
    )


@pytest.mark.parametrize(
    "input_file_or_dir, tokenizer_type, spm_split_by_whitespace",
    [
        (PRETOKENIZED_FILE, "sentencepiece", False),
        (PRETOKENIZED_FILE, "wordpiece", False),
        (PRETOKENIZED_DIR, "sentencepiece", True),
        (PRETOKENIZED_DIR, "wordpiece", False),
    ],
)
def test_train_tokenizer(
    input_file_or_dir: str,
    tokenizer_type: str,
    spm_split_by_whitespace: bool,
) -> None:
    output_dir: str = "output/"
    vocab_size: int = 800
    min_frequency: int = 2
    limit_alphabet: int = 300
    num_unused_tokens: int = 5
    language: str = "ja"
    train_tokenizer.train_tokenizer(
        input_file_or_dir=input_file_or_dir,
        output_dir=output_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        limit_alphabet=limit_alphabet,
        num_unused_tokens=num_unused_tokens,
        tokenizer_type=tokenizer_type,
        language=language,
        spm_split_by_whitespace=spm_split_by_whitespace,
    )
