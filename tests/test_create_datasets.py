import os
import sys

import pytest
from jptranstokenizer import JapaneseTransformerTokenizer

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
import create_datasets

# TODO: write tests for masking pattern
@pytest.mark.parametrize(
    "dataset_type, mask_style",
    [
        ["linebyline", "none"],
        ["nsp", "none"],
        ["", "bert"],
        ["", "roberta"],
        ["", "deberta-wwm"],
        ["", "electra"],
        ["", "electra-wwm"],
        ["linebyline", "bert"],
        # ["", "bert-wwm"],
    ],
)
def test_make_dataset(dataset_type: str, mask_style: str):
    tokenizer: JapaneseTransformerTokenizer = (
        JapaneseTransformerTokenizer.from_pretrained("izumi-lab/bert-small-japanese")
    )
    ds = create_datasets.make_dataset(
        input_corpus="test",
        input_file="data/botchan_ja.txt",
        dataset_type=dataset_type,
        mask_style=mask_style,
        tokenizer=tokenizer,
        max_length=128,
        do_save=True,
    )
