import collections
import copy
import os
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


# TODO: This function should be replaced with the new tokenizer
def load_tokenizer(
    tokenizer_name_or_path:str,
    tokenizer_type:str,
    mecab_dic_type:str
) -> PreTrainedTokenizerBase:

    if os.path.isfile(tokenizer_name_or_path+ "vocab.txt"):
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
            # tokenizer.enable_truncation(max_length = MAX_LENGTH)
            # convert to transformers style
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object = tokenizer,
                # model_max_length = MAX_LENGTH,
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
            tokenizer = BertJapaneseTokenizer(
                tokenizer_name_or_path,
                do_lower_case = False,
                word_tokenizer_type = "mecab",
                subword_tokenizer_type = "wordpiece",
                tokenize_chinese_chars = False,
                mecab_kwargs = {"mecab_dic": mecab_dic_type},
                # model_max_length = MAX_LENGTH
            )
        else:
            raise ValueError(f"Invalid tokenizer_type {tokenizer_type}.")
    else:
        # load from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    return tokenizer


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

        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []
        for mrph in self.juman.analysis(text):
            token = mrph.midasi
            if self.do_lower_case and token not in never_split:
                token = token.lower()
            tokens.append(token)
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
        word_tokenizer_type="basic",
        subword_tokenizer_type="wordpiece",
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        mecab_kwargs=None,
        sudachi_kwargs=None,
        sp_model_kwargs=None,
        call_from_pretrained=False,
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
            word_tokenizer_type=word_tokenizer_type,
            subword_tokenizer_type=subword_tokenizer_type,
            never_split=never_split,
            mecab_kwargs=mecab_kwargs,
            **kwargs,
        )
        # ^^ We call the grandparent's init, not the parent's.

        if not os.path.isfile(vocab_file) and not call_from_pretrained:
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        if subword_tokenizer_type != "sentencepiece" and not call_from_pretrained:
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        self.do_word_tokenize = do_word_tokenize
        self.word_tokenizer_type = word_tokenizer_type
        self.lower_case = do_lower_case
        self.never_split = never_split
        self.mecab_kwargs = copy.deepcopy(mecab_kwargs)
        if do_word_tokenize:
            self.word_tokenizer = get_word_tokenizer(
                word_tokenizer_type=word_tokenizer_type,
                do_lower_case=do_lower_case,
                never_split=never_split,
                mecab_kwargs=mecab_kwargs,
                sudachi_kwargs=sudachi_kwargs
            )

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if self.do_subword_tokenize and not call_from_pretrained:
            if subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif subword_tokenizer_type == "sentencepiece":
                self.subword_tokenizer = SentencePieceTokenizer(
                    vocab_file=vocab_file, sp_model_kwargs=sp_model_kwargs
                )
                self.vocab = self.subword_tokenizer.vocab
                self.ids_to_tokens = [self.subword_tokenizer.spm.IdToPiece(i) for i in range(self.subword_tokenizer.bpe_vocab_size)]
            else:
                raise ValueError(f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified.")
        # This is needed for leave special tokens as it is when tokenizing
        self.unique_no_split_tokens = list(self.special_tokens_map.values())
    
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], 
        word_tokenizer_type: str,
        tokenizer_class: str,
        do_lower_case=False,
        do_word_tokenize=True,
        never_split=None,
        mecab_kwargs=None,
        sudachi_kwargs=None,
        *init_inputs, **kwargs
    ):
        tokenizer_class = transformers.models.auto.tokenization_auto.tokenizer_class_from_name(tokenizer_class)
        tentative_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        if (
            isinstance(tentative_tokenizer, transformers.T5Tokenizer) or
            isinstance(tentative_tokenizer, transformers.AlbertTokenizer)
        ):
            # sentencepiece
            subword_tokenizer_type = "sentencepiece"
            subword_tokenizer = SentencePieceTokenizer(
                vocab_file=None,
                sp_model_kwargs=None,
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
            word_tokenizer_type=word_tokenizer_type,
            subword_tokenizer_type=subword_tokenizer_type,
            never_split=never_split,
            unk_token=tentative_tokenizer.special_tokens_map["unk_token"],
            sep_token=tentative_tokenizer.special_tokens_map["sep_token"],
            pad_token=tentative_tokenizer.special_tokens_map["pad_token"],
            cls_token=tentative_tokenizer.special_tokens_map["cls_token"],
            mask_token=tentative_tokenizer.special_tokens_map["mask_token"],
            mecab_kwargs=mecab_kwargs,
            sudachi_kwargs=sudachi_kwargs,
            call_from_pretrained=True,
        )
        tokenizer.subword_tokenizer = subword_tokenizer
        tokenizer.vocab = vocab
        tokenizer.ids_to_tokens = ids_to_tokens
        
        # This is needed for leave special tokens as it is when tokenizing
        tokenizer.unique_no_split_tokens = list(tokenizer.special_tokens_map.values())
        return tokenizer
