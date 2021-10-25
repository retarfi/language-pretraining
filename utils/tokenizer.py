import os

from tokenizers import SentencePieceBPETokenizer, normalizers
from tokenizers.processors import BertProcessing
import transformers
from transformers import (
    AutoTokenizer,
    BertJapaneseTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    logging
)

logging.set_verbosity_info()
logging.enable_explicit_format()
logger = logging.get_logger()


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