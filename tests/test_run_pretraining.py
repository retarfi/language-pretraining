import os
import random
import sys
from contextlib import nullcontext as does_not_raise
from typing import Dict, Union

import pytest
from jptranstokenizer import JapaneseTransformerTokenizer
from transformers.deepspeed import is_deepspeed_available

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
import run_pretraining


@pytest.mark.parametrize(
    "model_name, dataset_dirname, is_dataset_masked, expectation",
    [
        ["bert", "nsp_128_test", False, does_not_raise()],
        ["debertav2", "linebyline_128_test_deberta-wwm", True, does_not_raise()],
        ["electra", "linebyline_128_test_electra", True, does_not_raise()],
        ["electra", "linebyline_128_test_electra-wwm", True, does_not_raise()],
        ["roberta", "linebyline_128_test_roberta", True, does_not_raise()],
        ["roberta", "linebyline_128_test", False, does_not_raise()],
        ["bert", "hogefuga", True, pytest.raises(AssertionError)]
    ],
)
def test_run_pretraining(
    model_name: str, dataset_dirname: str, is_dataset_masked: bool, expectation
) -> None:
    with expectation:
        tokenizer: JapaneseTransformerTokenizer = (
            JapaneseTransformerTokenizer.from_pretrained("izumi-lab/bert-small-japanese")
        )
        params: Dict[str, Union[float, int, str]] = {
            "number-of-layers": 4,
            "hidden-size": 64,
            "sequence-length": 128,
            "ffn-inner-hidden-size": 128,
            "attention-heads": 4,
            "warmup-steps": 10,
            "learning-rate": 5e-4,
            "batch-size": {"-1": 8},
            "train-steps": 20,
            "save-steps": 15,
            "logging-steps": 5,
        }
        if model_name == "electra":
            params["embedding-size"] = 64
            params["generator-size"] = "1/4"
            params["mask-percent"] = 15
        use_deepspeed: bool
        if is_deepspeed_available() and random.random() > 0.5:
            use_deepspeed = True
        else:
            use_deepspeed = False
        run_pretraining.run_pretraining(
            tokenizer=tokenizer,
            dataset_dir=f"./dataset/{dataset_dirname}",
            model_name=model_name,
            model_dir="./model/",
            load_pretrained=False,
            param_config=params,
            do_whole_word_mask=("-wwm" in dataset_dirname),
            is_dataset_masked=is_dataset_masked,
            use_deepspeed=use_deepspeed,
        )
