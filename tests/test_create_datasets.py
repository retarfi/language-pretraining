import os
import sys

import pytest
from jptranstokenizer import JapaneseTransformerTokenizer

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
import create_datasets


@pytest.mark.parametrize("dataset_type", ["linebyline", "nsp"])
def test_make_dataset(dataset_type: str):
    tokenizer: JapaneseTransformerTokenizer = (
        JapaneseTransformerTokenizer.from_pretrained("izumi-lab/bert-small-japanese")
    )
    ds = create_datasets.make_dataset(
        input_corpus="test",
        input_file="data/botchan_ja.txt",
        dataset_type=dataset_type,
        tokenizer=tokenizer,
        max_length=128,
        do_save=True,
    )
