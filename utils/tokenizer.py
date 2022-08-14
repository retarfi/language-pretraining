import argparse
import collections
import copy
import os
import re
import unicodedata
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Union

from tokenizers import SentencePieceBPETokenizer, normalizers
from tokenizers.processors import BertProcessing
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertJapaneseTokenizer,
    BertTokenizer,
    DebertaV2Tokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    RobertaTokenizer,
    logging
)
from transformers.models.bert.tokenization_bert import BasicTokenizer, WordpieceTokenizer, load_vocab
from transformers.models.bert_japanese.tokenization_bert_japanese import CharacterTokenizer, MecabTokenizer


logging.set_verbosity_info()
logging.enable_explicit_format()
logger = logging.get_logger()


def load_tokenizer(args: argparse.Namespace) -> PreTrainedTokenizerBase:
    def _load_tokenizer(
        language: str,
        tokenizer_name_or_path: Union[str, os.PathLike],
        load_from_hub: bool,
        word_tokenizer: Optional[str],
        subword_tokenizer: Optional[str],
        tokenizer_class: Optional[str],
        do_lower_case: bool,
        do_word_tokenize: bool,
        never_split: Optional[List[str]],
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
                return JapaneseTokenizer.from_pretrained(
                    tokenizer_name_or_path=tokenizer_name_or_path, 
                    word_tokenizer=word_tokenizer,
                    tokenizer_class=tokenizer_class,
                    do_lower_case=do_lower_case,
                    do_word_tokenize=do_word_tokenize,
                    never_split=never_split,
                    mecab_dic=mecab_dic,
                    mecab_option=mecab_option,
                    sudachi_split_mode=sudachi_split_mode,
                    sudachi_config_path=sudachi_config_path,
                    sudachi_resource_dir=sudachi_resource_dir,
                    sudachi_dict_type=sudachi_resource_dir,
                    sp_model_kwargs=sp_model_kwargs
                )
            else:
                return JapaneseTokenizer(
                    vocab_file=tokenizer_name_or_path,
                    do_lower_case=do_lower_case,
                    do_word_tokenize=do_word_tokenize,
                    word_tokenizer=word_tokenizer,
                    subword_tokenizer=subword_tokenizer,
                    never_split=never_split,
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
                    sp_model_kwargs=sp_model_kwargs
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
        word_tokenizer=args.word_tokenizer,
        subword_tokenizer=args.subword_tokenizer,
        tokenizer_class=args.tokenizer_class,
        do_lower_case=args.do_lower_case,
        do_word_tokenize=args.do_word_tokenize,
        never_split=args.never_split,
        mecab_dic=args.mecab_dic,
        mecab_option=args.mecab_option,
        sudachi_split_mode=args.sudachi_split_mode,
        sudachi_config_path=args.sudachi_config_path,
        sudachi_resource_dir=args.resource_dir,
        sudachi_dict_type=args.sudachi_dict_type,
        sp_model_kwargs=args.sp_model_kwargs,
        unk_token=args.unk_token,
        sep_token=args.sep_token,
        pad_token=args.pad_token,
        cls_token=args.cls_token,
        mask_token=args.mask_token,
    )


def add_arguments_for_tokenizer(parser: argparse.Namespace) -> None:
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True, 
                        help="uploaded name in HuggingFace Hub or path to model or vocab file")
    parser.add_argument('--language', default='ja', choices=['ja', 'en'])
    parser.add_argument('--load_from_hub', action='store_true')
    parser.add_argument('--word_tokenizer', choices=['basic', 'mecab', 'juman', 'spacy-luw', 'sudachi', 'none', None])
    parser.add_argument('--subword_tokenizer', choices=['wordpiece', 'sentencepiece', None])
    parser.add_argument('--tokenizer_class')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--do_word_tokenize', action='store_true')
    parser.add_argument('--never_split', nargs='*')
    parser.add_argument('--mecab_dic')
    parser.add_argument('--mecab_option')
    parser.add_argument('--sudachi_split_mode')
    parser.add_argument('--sudachi_config_path')
    parser.add_argument('--sudachi_resource_dir')
    parser.add_argument('--sudachi_dict_type', default='core')
    parser.add_argument('--sp_model_kwargs')
    parser.add_argument('--unk_token', default='[UNK]')
    parser.add_argument('--sep_token', default='[SEP]')
    parser.add_argument('--pad_token', default='[PAD]')
    parser.add_argument('--cls_token', default='[CLS]')
    parser.add_argument('--mask_token', default='[MASK]')


def assert_arguments_for_tokenizer(args: argparse.Namespace) -> None:
    # confirm validation of arguments with load_tokenizer()
    assert args.tokenizer_name_or_path != "", "Argument tokenizer_name_or_path must be specified"
    if args.language == "ja":
        assert args.word_tokenizer is not None, "Argument word_tokenizer must be explicitly specified (basic, mecab, juman, spacy-luw, sudachi, none)"
        if args.load_from_hub:
            assert args.tokenizer_class is not None, "Argument tokenizer_class must be specified"
        else:
            assert args.subword_tokenizer is not None, "Argument subword_tokenizer must be specified (wordpiece, sentencepiece)"
    elif args.language == "en":
        if args.load_from_hub:
            pass
        else:
            logger.error("the mode of language: en and loading local tokenizer is not implemented now")
            raise NotImplementedError()
    else:
        raise ValueError('Argument language must be "ja" or "en') 


class MainTokenizerABC(ABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            **do_lower_case**: (*optional*) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
            **normalize_text**: (*optional*) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text

    @abstractmethod
    def tokenize(
        self,
        text: str,
        never_split: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        pass


class Normalizer(MainTokenizerABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True
    ):
        super().__init__(
            do_lower_case = False,
            never_split = None,
            normalize_text = True
        )

    def tokenize(
        self,
        text: str,
        never_split: Optional[List[str]] = None
    ) -> List[str]:
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)
        never_split = self.never_split + (never_split if never_split is not None else [])
        return text


class JumanTokenizer(MainTokenizerABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True
    ):
        super().__init__(
            do_lower_case = False,
            never_split = None,
            normalize_text = True
        )
        try:
            from pyknp import Juman
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install pyknp to use JumanTokenizer."
                "See https://github.com/ku-nlp/pyknp for installation."
            )
        self.juman = Juman()

    def tokenize(
        self,
        text: str,
        never_split: Optional[List[str]] = None
    ) -> List[str]:
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)
        # "#" and "@" at the beginning of a sentence causes timeout error
        text = re.sub("^#", "＃", text)
        text = re.sub("^@", "＠", text)
        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []
        try:
            result = self.juman.analysis(text)
            use_underscore = False
            use_quote = False
        except ValueError:
            # This error is occured because of the Juman's matter about space
            if '"' in text:
                text = text.replace('"', '”')
                use_quote = True
            else:
                use_quote = False
            if re.search("\s", text):
                text = re.sub("\s", "_", text)
                use_underscore = True
            else:
                use_underscore = False
            try:
                result = self.juman.analysis(text)
            except Exception:
                print(text)
                import sys
                sys.exit(1)
        except Exception:
            print(text)
            import sys
            sys.exit(1)
        for mrph in result:
            token = mrph.midasi
            if self.do_lower_case and token not in never_split:
                token = token.lower()
            tokens.append(token)
        if use_underscore:
            tokens = list(filter(lambda x: x != "_", tokens))
        if use_quote:
            tokens = list(map(lambda x: x.replace('”', '"'), tokens))
        return tokens


class SpacyluwTokenizer(MainTokenizerABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True
    ):
        super().__init__(
            do_lower_case = False,
            never_split = None,
            normalize_text = True
        )
        try:
            import spacy
            self.nlp = spacy.load("ja_gsdluw")
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install spacy to use SpacyluwTokenizer."
                "See https://pypi.org/project/spacy/ for spacy installation "
                "Also you would need to install ja_gsdluw https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE for ja_gsdluw installation."
            )
        except OSError as error:
            raise error.__class__(
                "You need to install ja_gsdluw to use SpacyluwTokenizer."
                "See https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE for installation."
            )
        
    def tokenize(
        self,
        text: str,
        never_split: Optional[List[str]] = None
    ) -> List[str]:
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []
        doc = self.nlp(text)
        tokens = [token.text for sent in doc.sents for token in sent]
        if self.do_lower_case:
            tokens = [token if token in never_split else token.lower() for token in tokens]
        return tokens


class SudachiTokenizer(MainTokenizerABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True,
        split_mode: Optional[str] = "A",
        config_path: Optional[str] = None,
        resource_dir: Optional[str] = None,
        dict_type: Optional[str] = "core",
    ):
        super().__init__(
            do_lower_case = False,
            never_split = None,
            normalize_text = True
        )
        try:
            from sudachitra.word_formatter import word_formatter
            from sudachitra.sudachipy_word_tokenizer import SudachipyWordTokenizer
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install sudachitra to use SudachipyWordTokenizer."
                "See https://pypi.org/project/SudachiTra/ for installation."
            )
            # cf. https://pypi.org/project/SudachiTra/
            # cf. https://github.com/WorksApplications/SudachiTra/blob/main/sudachitra/tokenization_bert_sudachipy.py
            # cf. https://github.com/WorksApplications/SudachiTra/blob/main/sudachitra/sudachipy_word_tokenizer.py
        self.sudachi_tokenizer = SudachipyWordTokenizer(
            split_mode=split_mode,
            config_path=config_path,
            resource_dir=resource_dir,
            dict_type=dict_type
        )
        self.word_formatter = word_formatter("surface", self.sudachi_tokenizer.sudachi_dict)
    
    def tokenize(
        self,
        text: str,
        never_split: Optional[List[str]] = None
    ) -> List[str]:
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = [self.word_formatter(token) for token in self.sudachi_tokenizer.tokenize(text)]
        if self.do_lower_case:
            tokens = [token if token in never_split else token.lower() for token in tokens]
        return tokens


class SentencePieceTokenizer:
    def __init__(self, vocab_file, sp_model_kwargs, spm=None):
        if spm is None:
            import sentencepiece as sp
            self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
            self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
            self.spm.load(vocab_file)
        else:
            self.spm = spm
        self.bpe_vocab_size = self.spm.GetPieceSize()
        self.vocab = {self.spm.IdToPiece(i): i for i in range(self.bpe_vocab_size)}
    
    def tokenize(
        self,
        text: str,
    ) -> List[str]:
        tokens = self.spm.encode_as_pieces(text)
        return tokens


def get_word_tokenizer(
    word_tokenizer_type: str,
    do_lower_case: bool,
    never_split: Optional[List[str]] = None,
    normalize_text: bool = True,
    mecab_dic: Optional[str] = "ipadic",
    mecab_option: Optional[str] = None,
    sudachi_split_mode: Optional[str] = "A",
    sudachi_config_path: Optional[str] = None,
    sudachi_resource_dir: Optional[str] = None,
    sudachi_dict_type: Optional[str] = "core",
):
    if word_tokenizer_type == "basic":
        logger.warn("Argument normalize_text is ignored")
        word_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=False
        )
    elif word_tokenizer_type == "mecab":
        word_tokenizer = MecabTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text,
            mecab_dic=mecab_dic,
            mecab_option=mecab_option
        )
    elif word_tokenizer_type == "juman":
        word_tokenizer = JumanTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text
        )
    elif word_tokenizer_type == "spacy-luw":
        word_tokenizer = SpacyluwTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text
        )
    elif word_tokenizer_type == "sudachi":
        word_tokenizer = SudachiTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text,
            split_mode=sudachi_split_mode,
            config_path=sudachi_config_path,
            resource_dir=sudachi_resource_dir,
            dict_type=sudachi_dict_type,
        )
    elif word_tokenizer_type == "none":
        word_tokenizer = Normalizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text
        )
    else:
        raise ValueError(f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified.")
    return word_tokenizer


class JapaneseTokenizer(BertJapaneseTokenizer):
    def __init__(
        self,
        vocab_file="",
        do_lower_case=False,
        do_word_tokenize=True,
        do_subword_tokenize=True,
        word_tokenizer="basic",
        subword_tokenizer="wordpiece",
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        call_from_pretrained=False,
        mecab_dic: Optional[str] = "ipadic",
        mecab_option: Optional[str] = None,
        sudachi_split_mode: Optional[str] = "A",
        sudachi_config_path: Optional[str] = None,
        sudachi_resource_dir: Optional[str] = None,
        sudachi_dict_type: Optional[str] = "core",
        sp_model_kwargs: Optional[str] = None,
        **kwargs
    ):
        super(BertTokenizer, self).__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=do_subword_tokenize,
            word_tokenizer=word_tokenizer,
            subword_tokenizer=subword_tokenizer,
            never_split=never_split,
            **kwargs,
        )
        # ^^ We call the grandparent's init, not the parent's.

        if not os.path.isfile(vocab_file) and not call_from_pretrained:
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        if subword_tokenizer != "sentencepiece" and not call_from_pretrained:
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        self.do_word_tokenize = do_word_tokenize
        self.lower_case = do_lower_case
        self.never_split = never_split
        if do_word_tokenize:
            self.word_tokenizer = get_word_tokenizer(
                word_tokenizer_type=word_tokenizer,
                do_lower_case=do_lower_case,
                never_split=never_split,
                mecab_dic=mecab_dic,
                mecab_option=mecab_option,
                sudachi_split_mode=sudachi_split_mode,
                sudachi_config_path=sudachi_config_path,
                sudachi_resource_dir=sudachi_resource_dir,
                sudachi_dict_type=sudachi_dict_type
            )

        self.do_subword_tokenize = do_subword_tokenize
        if self.do_subword_tokenize and not call_from_pretrained:
            if subword_tokenizer == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif subword_tokenizer == "character":
                self.subword_tokenizer = CharacterTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif subword_tokenizer == "sentencepiece":
                self.subword_tokenizer = SentencePieceTokenizer(
                    vocab_file=vocab_file, sp_model_kwargs=sp_model_kwargs
                )
                self.vocab = self.subword_tokenizer.vocab
                self.ids_to_tokens = [self.subword_tokenizer.spm.IdToPiece(i) for i in range(self.subword_tokenizer.bpe_vocab_size)]
            else:
                raise ValueError(f"Invalid subword_tokenizer '{subword_tokenizer}' is specified.")
        # This is needed for leave special tokens as it is when tokenizing
        self.unique_no_split_tokens = list(self.special_tokens_map.values())
    
    @classmethod
    def from_pretrained(
        cls,
        tokenizer_name_or_path: Union[str, os.PathLike], 
        word_tokenizer: str,
        tokenizer_class: str,
        do_lower_case=False,
        do_word_tokenize=True,
        never_split=None,
        mecab_dic: Optional[str] = "ipadic",
        mecab_option: Optional[str] = None,
        sudachi_split_mode: Optional[str] = "A",
        sudachi_config_path: Optional[str] = None,
        sudachi_resource_dir: Optional[str] = None,
        sudachi_dict_type: Optional[str] = "core",
        sp_model_kwargs: Optional[str] = None,
        *init_inputs, **kwargs
    ):
        tokenizer_class = transformers.models.auto.tokenization_auto.tokenizer_class_from_name(tokenizer_class)
        tentative_tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, *init_inputs, **kwargs)
        if (
            isinstance(tentative_tokenizer, transformers.T5Tokenizer) or
            isinstance(tentative_tokenizer, transformers.AlbertTokenizer)
        ):
            # sentencepiece
            subword_tokenizer_type = "sentencepiece"
            subword_tokenizer = SentencePieceTokenizer(
                vocab_file=None,
                sp_model_kwargs=sp_model_kwargs,
                spm=tentative_tokenizer.sp_model
            )
            vocab = subword_tokenizer.vocab
            ids_to_tokens = [
                subword_tokenizer.spm.IdToPiece(i) for i in range(subword_tokenizer.bpe_vocab_size)
            ]
        elif isinstance(tentative_tokenizer, BertJapaneseTokenizer):
            # WordPiece or character
            subword_tokenizer = tentative_tokenizer.subword_tokenizer
            if isinstance(subword_tokenizer, WordpieceTokenizer):
                subword_tokenizer_type = "wordpiece"
            elif isinstance(subword_tokenizer, CharacterTokenizer):
                subword_tokenizer_type = "character"
            else:
                raise ValueError()
            vocab = tentative_tokenizer.vocab
            ids_to_tokens = tentative_tokenizer.ids_to_tokens
        else:
            raise NotImplementedError()
        tokenizer = cls(
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=True,
            word_tokenizer=word_tokenizer,
            subword_tokenizer=subword_tokenizer_type,
            never_split=never_split,
            unk_token=tentative_tokenizer.special_tokens_map["unk_token"],
            sep_token=tentative_tokenizer.special_tokens_map["sep_token"],
            pad_token=tentative_tokenizer.special_tokens_map["pad_token"],
            cls_token=tentative_tokenizer.special_tokens_map["cls_token"],
            mask_token=tentative_tokenizer.special_tokens_map["mask_token"],
            call_from_pretrained=True,
            mecab_dic=mecab_dic,
            mecab_option=mecab_option,
            sudachi_split_mode=sudachi_split_mode,
            sudachi_config_path=sudachi_config_path,
            sudachi_resource_dir=sudachi_resource_dir,
            sudachi_dict_type=sudachi_dict_type
        )
        tokenizer.subword_tokenizer = subword_tokenizer
        tokenizer.vocab = vocab
        tokenizer.ids_to_tokens = ids_to_tokens
        
        # This is needed for leave special tokens as it is when tokenizing
        tokenizer.unique_no_split_tokens = list(tokenizer.special_tokens_map.values())
        return tokenizer
