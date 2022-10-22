import argparse
import os
from typing import Optional, Union

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    logging,
)
from jptranstokenizer import JapaneseTransformerTokenizer


logging.set_verbosity_info()
logging.enable_explicit_format()
logger = logging.get_logger()


def load_tokenizer(args: argparse.Namespace) -> PreTrainedTokenizerBase:
    def _load_tokenizer(
        language: str,
        tokenizer_name_or_path: Union[str, os.PathLike],
        load_from_hub: bool,
        word_tokenizer_type: Optional[str],
        subword_tokenizer_type: Optional[str],
        tokenizer_class: Optional[str],
        do_lower_case: bool,
        ignore_max_byte_error: bool,
        mecab_dic: Optional[str],
        mecab_option: Optional[str],
        sudachi_split_mode: Optional[str],
        sudachi_config_path: Optional[str],
        sudachi_resource_dir: Optional[str],
        sudachi_dict_type: Optional[str],
        sp_model_kwargs: Optional[str],
        unk_token: str,
        sep_token: str,
        pad_token: str,
        cls_token: str,
        mask_token: str,
    ) -> PreTrainedTokenizerBase:
        if language == "ja":
            if load_from_hub:
                return JapaneseTransformerTokenizer.from_pretrained(
                    tokenizer_name_or_path=tokenizer_name_or_path,
                    word_tokenizer_type=word_tokenizer_type,
                    tokenizer_class=tokenizer_class,
                    do_lower_case=do_lower_case,
                    ignore_max_byte_error=ignore_max_byte_error,
                    mecab_dic=mecab_dic,
                    mecab_option=mecab_option,
                    sudachi_split_mode=sudachi_split_mode,
                    sudachi_config_path=sudachi_config_path,
                    sudachi_resource_dir=sudachi_resource_dir,
                    sudachi_dict_type=sudachi_dict_type,
                    sp_model_kwargs=sp_model_kwargs,
                )
            else:
                return JapaneseTransformerTokenizer(
                    vocab_file=tokenizer_name_or_path,
                    word_tokenizer_type=word_tokenizer_type,
                    subword_tokenizer_type=subword_tokenizer_type,
                    do_lower_case=do_lower_case,
                    ignore_max_byte_error=ignore_max_byte_error,
                    unk_token=unk_token,
                    sep_token=sep_token,
                    pad_token=pad_token,
                    cls_token=cls_token,
                    mask_token=mask_token,
                    mecab_dic=mecab_dic,
                    mecab_option=mecab_option,
                    sudachi_split_mode=sudachi_split_mode,
                    sudachi_config_path=sudachi_config_path,
                    sudachi_resource_dir=sudachi_resource_dir,
                    sudachi_dict_type=sudachi_resource_dir,
                    sp_model_kwargs=sp_model_kwargs,
                )
        elif language == "en":
            if load_from_hub:
                return AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            else:
                # TODO: Add implementation for english local tokenizer
                raise NotImplementedError()
        else:
            raise ValueError("language must be ja or en")

    return _load_tokenizer(
        language=args.language,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        load_from_hub=args.load_from_hub,
        word_tokenizer_type=args.word_tokenizer_type,
        subword_tokenizer_type=args.subword_tokenizer_type,
        tokenizer_class=args.tokenizer_class,
        do_lower_case=args.do_lower_case,
        ignore_max_byte_error=args.ignore_max_byte_error,
        mecab_dic=args.mecab_dic,
        mecab_option=args.mecab_option,
        sudachi_split_mode=args.sudachi_split_mode,
        sudachi_config_path=args.sudachi_config_path,
        sudachi_resource_dir=args.sudachi_resource_dir,
        sudachi_dict_type=args.sudachi_dict_type,
        sp_model_kwargs=args.sp_model_kwargs,
        unk_token=args.unk_token,
        sep_token=args.sep_token,
        pad_token=args.pad_token,
        cls_token=args.cls_token,
        mask_token=args.mask_token,
    )


def add_arguments_for_tokenizer(parser: argparse.Namespace) -> None:
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        help="uploaded name in HuggingFace Hub or path to model or vocab file",
    )
    parser.add_argument("--language", default="ja", choices=["ja", "en"])
    parser.add_argument("--load_from_hub", action="store_true")
    parser.add_argument(
        "--word_tokenizer_type",
        choices=["basic", "mecab", "juman", "spacy-luw", "sudachi", "none", None],
    )
    parser.add_argument(
        "--subword_tokenizer_type", choices=["wordpiece", "sentencepiece", None]
    )
    parser.add_argument("--tokenizer_class")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--ignore_max_byte_error", action="store_true")
    parser.add_argument("--mecab_dic")
    parser.add_argument("--mecab_option")
    parser.add_argument("--sudachi_split_mode")
    parser.add_argument("--sudachi_config_path")
    parser.add_argument("--sudachi_resource_dir")
    parser.add_argument("--sudachi_dict_type", default="core")
    parser.add_argument("--sp_model_kwargs")
    parser.add_argument("--unk_token", default="[UNK]")
    parser.add_argument("--sep_token", default="[SEP]")
    parser.add_argument("--pad_token", default="[PAD]")
    parser.add_argument("--cls_token", default="[CLS]")
    parser.add_argument("--mask_token", default="[MASK]")


def assert_arguments_for_tokenizer(args: argparse.Namespace) -> None:
    # confirm validation of arguments with load_tokenizer()
    assert (
        args.tokenizer_name_or_path != ""
    ), "Argument tokenizer_name_or_path must be specified"
    if args.language == "ja":
        if not args.load_from_hub:
            assert (
                args.word_tokenizer_type is not None
            ), "Argument word_tokenizer must be explicitly specified (basic, mecab, juman, spacy-luw, sudachi, none)"
            assert (
                args.subword_tokenizer_type is not None
            ), "Argument subword_tokenizer must be specified (wordpiece, sentencepiece)"
    elif args.language == "en":
        if args.load_from_hub:
            pass
        else:
            logger.error(
                "the mode of language: en and loading local tokenizer is not implemented now"
            )
            raise NotImplementedError()
    else:
        raise ValueError('Argument language must be "ja" or "en')
