import os
import sys
from contextlib import nullcontext as does_not_raise

import pytest
from jptranstokenizer import JapaneseTransformerTokenizer

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
import create_datasets

@pytest.mark.parametrize(
    "dataset_type, mask_style, expectation",
    [
        ["linebyline", "none", does_not_raise()],
        ["nsp", "none", does_not_raise()],
        ["", "roberta", does_not_raise()],
        ["", "deberta-wwm", does_not_raise()],
        ["", "electra", does_not_raise()],
        ["", "electra-wwm", does_not_raise()],
        ["", "bert", pytest.raises(AssertionError)],
        ["", "bert-wwm", pytest.raises(AssertionError)]
    ],
)
def test_make_dataset(dataset_type: str, mask_style: str, expectation):
    with expectation:
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
