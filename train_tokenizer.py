"""
Pre-tokenize用にファイルを分割する。
MeCabでPre-tokenizeしたのちsubword用にtokenizerをtrainする。
"""
import argparse
import logging
import multiprocessing as mp
import os
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from jptranstokenizer import get_word_tokenizer

from utils.logger import make_logger_setting


# logger
logger: logging.Logger = logging.getLogger(__name__)
make_logger_setting(logger)
BAR_FORMAT: str = "{n_fmt}/{total_fmt}: {percentage:3.0f}%, [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def split(
    input_file: str, num_files: int, intermediate_dir: str, use_tqdm: bool = True
) -> None:
    if num_files > 1:
        logger.info("Splitting...")
        line_all: int = int(
            subprocess.run(
                ["wc", "-l", input_file],
                encoding="utf-8",
                stdout=subprocess.PIPE,
            ).stdout.split()[0]
        )
        intermediate_plib_dir: Path = Path(intermediate_dir)
        line_per_file: int = line_all // num_files + 5
        if use_tqdm:
            pbar: tqdm = tqdm(total=num_files, bar_format=BAR_FORMAT)
        cnt_file: int = 0
        cnt_line: int = 0
        os.makedirs(intermediate_dir, exist_ok=True)
        with open(input_file, "r") as infile:
            f = open((intermediate_plib_dir / f"{cnt_file}.txt").resolve(), "w")
            for line in infile:
                if cnt_line >= line_per_file and line == "\n":
                    f.write("\n")
                    f.close()
                    if use_tqdm:
                        pbar.update(1)
                    cnt_file += 1
                    f = open((intermediate_plib_dir / f"{cnt_file}.txt").resolve(), "w")
                    cnt_line = 0
                else:
                    f.write(line)
                    cnt_line += 1
            f.close()
            if use_tqdm:
                pbar.update(1)
                pbar.close()
        assert num_files == cnt_file + 1


def mp_tokenize(
    input_txt: str,
    output_txt: str,
    num_file: int,
    word_tokenizer_type: str,
    mecab_dic: str = "",
    mecab_option: str = "",
    sudachi_split_mode: Optional[str] = None,
    sudachi_config_path: Optional[str] = None,
    sudachi_resource_dir: Optional[str] = None,
    sudachi_dict_type: Optional[str] = None,
    use_tqdm: bool = True,
    ignore_max_byte_error: bool = False,
    ignore_runtime_error: bool = False,
    delimiter: str = " ",
) -> None:
    # ignore_runtime_error is option for spacy-luw

    main_tokenizer = get_word_tokenizer(
        word_tokenizer_type=word_tokenizer_type,
        do_lower_case=False,
        mecab_dic=mecab_dic,
        mecab_option=mecab_option,
        sudachi_split_mode=sudachi_split_mode,
        sudachi_config_path=sudachi_config_path,
        sudachi_resource_dir=sudachi_resource_dir,
        sudachi_dict_type=sudachi_dict_type,
        ignore_max_byte_error=ignore_max_byte_error,
    )
    if num_file == 0 and use_tqdm:
        line_all = int(
            subprocess.run(
                ["wc", "-l", input_txt], encoding="utf-8", stdout=subprocess.PIPE
            ).stdout.split()[0]
        )
        pbar: tqdm = tqdm(total=line_all, bar_format=BAR_FORMAT)
    with open(input_txt, "r") as infile, open(output_txt, "w") as outfile:
        for line in infile:
            if line == "\n":
                outfile.write("\n")
            else:
                if ignore_runtime_error:
                    try:
                        outfile.write(
                            delimiter.join(main_tokenizer.tokenize(line.strip())) + "\n"
                        )
                    except RuntimeError:
                        pass
                else:
                    outfile.write(
                        delimiter.join(main_tokenizer.tokenize(line.strip())) + "\n"
                    )
            if num_file == 0 and use_tqdm:
                pbar.update(1)
    if num_file == 0 and use_tqdm:
        pbar.close()


def pre_tokenize(
    input_file: str,
    num_files: int,
    pretokenized_prefix: str,
    intermediate_dir: str,
    word_tokenizer: str,
    mecab_dic: str,
    mecab_option: str,
    sudachi_split_mode: str,
    sudachi_config_path: str,
    sudachi_resource_dir: str,
    sudachi_dict_type: str,
    use_tqdm: bool = True,
    ignore_max_byte_error: bool = False,
    delimiter: str = " ",
) -> str:
    logger.info("Pre-tokenizing...")
    input_file_or_dir: str
    if num_files == 1:
        input_plib_file: Path = Path(input_file)
        pretokenized_plib_file = input_plib_file.parent.joinpath(
            input_plib_file.stem + pretokenized_prefix + ".txt"
        )
        mp_tokenize(
            input_txt=str(input_plib_file),
            output_txt=str(pretokenized_plib_file),
            num_file=0,
            word_tokenizer_type=word_tokenizer,
            mecab_dic=mecab_dic,
            mecab_option=mecab_option,
            sudachi_split_mode=sudachi_split_mode,
            sudachi_config_path=sudachi_config_path,
            sudachi_resource_dir=sudachi_resource_dir,
            sudachi_dict_type=sudachi_dict_type,
            use_tqdm=use_tqdm,
            ignore_max_byte_error=ignore_max_byte_error,
            delimiter=delimiter,
        )
        logger.info(f"Pre-tokenized files are saved in {str(pretokenized_plib_file)}")
        input_file_or_dir = str(pretokenized_plib_file)
    else:
        intermediate_plib_dir: Path = Path(intermediate_dir)
        pretokenized_plib_dir: Path = intermediate_plib_dir.parent.joinpath(
            intermediate_plib_dir.stem + pretokenized_prefix
        )
        os.makedirs(pretokenized_plib_dir, exist_ok=True)
        with mp.Pool(num_files) as pool:
            mp_task = [
                pool.apply_async(
                    mp_tokenize,
                    (
                        str((intermediate_plib_dir / f"{i}.txt").resolve()),
                        str((pretokenized_plib_dir / f"{i}.txt").resolve()),
                        i,
                        word_tokenizer,
                        mecab_dic,
                        mecab_option,
                        sudachi_split_mode,
                        sudachi_config_path,
                        sudachi_resource_dir,
                        sudachi_dict_type,
                        use_tqdm,
                        ignore_max_byte_error,
                        delimiter,
                    ),
                )
                for i in range(num_files)
            ]
            _ = [f.get() for f in mp_task]
        logger.info(f"Pre-tokenized files are saved in {str(pretokenized_plib_dir)}")
        input_file_or_dir = str(pretokenized_plib_dir)
    return input_file_or_dir


def train_tokenizer(
    input_file_or_dir: str,
    output_dir: str,
    vocab_size: int,
    min_frequency: int,
    limit_alphabet: int,
    num_unused_tokens: int,
    tokenizer_type: str,
    language: str,
    spm_model_type: str = "unigram",
    spm_split_by_whitespace: bool = False,
    spm_add_dummy_prefix: bool = True,
    spm_delimiter: str = " ",
) -> None:

    if os.path.isfile(input_file_or_dir):
        files = [input_file_or_dir]
    elif os.path.isdir(input_file_or_dir):
        files = list(map(str, Path(input_file_or_dir).glob("*.txt")))
    else:
        raise ValueError(
            "argument input_file_or_dir must be text file or directory which consists .txt files."
        )

    logger.info("Train tokenizer...")
    os.makedirs(output_dir, exist_ok=True)
    if tokenizer_type == "sentencepiece":
        special_tokens = ["<unused{}>".format(i) for i in range(num_unused_tokens)]
        import sentencepiece as spm

        spm.SentencePieceTrainer.Train(
            input=files,
            model_type=spm_model_type,
            split_by_whitespace=spm_split_by_whitespace,
            add_dummy_prefix=spm_add_dummy_prefix,
            vocab_size=vocab_size,
            model_prefix=os.path.join(output_dir, "spiece"),
            character_coverage=0.9995,
            num_threads=min(os.cpu_count(), 128),
            train_extremely_large_corpus=True,
            pad_piece="[PAD]",
            unk_piece="[UNK]",
            bos_piece="[CLS]",
            eos_piece="[SEP]",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            control_symbols=["[MASK]"],
            user_defined_symbols=",".join(special_tokens),
            pretokenization_delimiter="" if spm_delimiter == " " else spm_delimiter,
        )
    elif tokenizer_type == "wordpiece":
        if language == "ja":
            tokenizer = BertWordPieceTokenizer(
                handle_chinese_chars=False, strip_accents=False, lowercase=False
            )
        elif language == "en":
            tokenizer = BertWordPieceTokenizer(
                handle_chinese_chars=True,
                strip_accents=None,  # determined by the value for lowercase
                lowercase=True,
            )
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        special_tokens += ["<unused{}>".format(i) for i in range(num_unused_tokens)]
        tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            special_tokens=special_tokens,
        )
        # save tokenizer
        tokenizer.save_model(output_dir)
    else:
        raise ValueError(f"Invalid tokenizer_type {tokenizer_type}.")
    logger.info("Tokenizer saved.")


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--word_tokenizer",
        required=True,
        type=str,
        choices=["mecab", "juman", "sudachi", "spacy-luw", "none", "basic"],
    )
    parser.add_argument(
        "--input_file",
        required=True,
        type=str,
        help="In the text file, new line must be inserted between two sentences. "
        "(For NSP) break line is inserted between two paragraphs.",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Directory for trained tokenizer model.",
    )
    parser.add_argument("--language", type=str, default="ja", choices=["ja", "en"])
    # parallel option
    parser.add_argument(
        "--intermediate_dir",
        type=str,
        default="tmp",
        help="Temporary directory for unpretokenized splitted texts.",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=1,
        help="Number of split files. It enables multiprocessing. "
        "Using multiprocessing with spacy-luw will not work.",
    )
    # pre-tokenize(mainword) option
    parser.add_argument(
        "--pretokenized_prefix",
        type=str,
        default="_pretokenized",
        help="Prefix of the directory of pretokenized texts",
    )
    # subword training option
    parser.add_argument(
        "--tokenizer_type",
        required=True,
        type=str,
        choices=["sentencepiece", "wordpiece"],
        help="Subword tokenizer type",
    )
    parser.add_argument(
        "--spm_model_type",
        type=str,
        choices=["unigram", "bpe", "char"],
        default="unigram",
        help="Sntencepiece model type",
    )
    parser.add_argument(
        "--spm_split_by_whitespace",
        action="store_true",
        help="If enabled, use a white space to split sentence pieces (only for sentencepiece)",
    )
    parser.add_argument(
        "--spm_pretokenize_boundary_constraint",
        action="store_true",
        help="If enabled, train the model with pre-tokenization boundary constraints (only for sentencepiece)",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32768, help="The number of vocabulary."
    )
    parser.add_argument("--min_frequency", type=int, default=2, help="only wordpiece")
    parser.add_argument(
        "--limit_alphabet", type=int, default=2900, help="only wordpiece"
    )
    parser.add_argument(
        "--num_unused_tokens",
        type=int,
        default=10,
        help="The number of vocabulary of unused tokens such as <unused0>.",
    )
    # mecab option
    parser.add_argument(
        "--mecab_dic",
        type=str,
        default="",
        choices=["", "unidic_lite", "unidic", "ipadic"],
        help="MeCab dict type. "
        "It must be specified when using MeCab for word tokenizer. "
        "For detail, please see jptranstokenizer's document.",
    )
    parser.add_argument(
        "--mecab_option",
        type=str,
        default="",
        help="It may be specified when using MeCab for word tokenizer. "
        "For detail, please see jptranstokenizer's document.",
    )
    # sudachi option
    parser.add_argument(
        "--sudachi_split_mode",
        default="",
        choices=["A", "B", "C", ""],
        help="It must be specified when using Sudachi for word tokenizer. "
        "For detail, please see jptranstokenizer's document.",
    )
    parser.add_argument(
        "--sudachi_config_path",
        help="It may be specified when using Sudachi for word tokenizer. "
        "For detail, please see jptranstokenizer's document.",
    )
    parser.add_argument(
        "--sudachi_resource_dir",
        help="It may be specified when using Sudachi for word tokenizer. "
        "For detail, please see jptranstokenizer's document.",
    )
    parser.add_argument(
        "--sudachi_dict_type",
        help="It may be specified when using Sudachi for word tokenizer. "
        "If not specified, it uses jptranstokenizer's default value (core)."
        "For detail, please see jptranstokenizer's document.",
    )
    # other option
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument(
        "--ignore_max_byte_error",
        action="store_true",
        help="This option enalbes to skip examples which would cause max byte error "
        "(for Juman++ and Sudachi)."
        "Please see get_word_tokenizer document of jptranstokenizer library.",
    )
    parser.add_argument(
        "--spm_disable_add_dummy_prefix",
        action="store_true",
        help="Disable adding dummy whitespace at the beginning of text (for sentencepiece)",
    )
    args = parser.parse_args()

    # assertion
    if args.language == "ja":
        assert args.word_tokenizer != "basic"
    elif args.language == "en":
        assert args.word_tokenizer in ["none", "basic"]
    else:
        raise ValueError("Invalid argument language")
    if ".txt" not in args.input_file:
        raise ValueError("input_file must be a txt file")
    if args.num_files < 1:
        raise ValueError("argument num_files must be 1 or larger")
    if args.word_tokenizer == "spacy-luw" and args.num_files > 1:
        logger.warn(
            "spacy-luw must be used with num_files==1, so changed num_files to 1"
        )
        args.num_files = 1
    if (
        args.tokenizer_type != "sentencepiece"
        and args.spm_pretokenize_boundary_constraint
    ):
        logger.warning(
            f"spm_pretokenize_boundary_constraint is not available with {args.tokenizer_type}.\n"
            "spm_pretokenize_boundary_constraint is disabled"
        )
        args.spm_pretokenize_boundary_constraint = False
    if args.spm_split_by_whitespace and args.spm_pretokenize_boundary_constraint:
        raise ValueError(
            "spm_split_by_whitespace and spm_pretokenize_boundary_constraint must not be specified at the same time"
        )
    delimiter: str
    if args.spm_pretokenize_boundary_constraint:
        delimiter = "||||"
    else:
        delimiter = " "

    use_tqdm: bool = not args.disable_tqdm
    input_file_or_dir: str
    if args.language == "ja":
        split(
            input_file=args.input_file,
            num_files=args.num_files,
            intermediate_dir=args.intermediate_dir,
            use_tqdm=use_tqdm,
        )

        input_file_or_dir = pre_tokenize(
            input_file=args.input_file,
            num_files=args.num_files,
            pretokenized_prefix=args.pretokenized_prefix,
            intermediate_dir=args.intermediate_dir,
            word_tokenizer=args.word_tokenizer,
            mecab_dic=args.mecab_dic,
            mecab_option=args.mecab_option,
            sudachi_split_mode=args.sudachi_split_mode,
            sudachi_config_path=args.sudachi_config_path,
            sudachi_resource_dir=args.sudachi_resource_dir,
            sudachi_dict_type=args.sudachi_dict_type,
            use_tqdm=use_tqdm,
            ignore_max_byte_error=args.ignore_max_byte_error,
            delimiter=delimiter,
        )
    else:
        input_file_or_dir = args.input_file

    train_tokenizer(
        input_file_or_dir=input_file_or_dir,
        output_dir=args.model_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet,
        num_unused_tokens=args.num_unused_tokens,
        tokenizer_type=args.tokenizer_type,
        language=args.language,
        spm_split_by_whitespace=args.spm_split_by_whitespace,
        spm_delimiter=delimiter,
        spm_add_dummy_prefix=not args.spm_disable_add_dummy_prefix,
    )
