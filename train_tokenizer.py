'''
Pre-tokenize用にファイルを分割する。
MeCabでPre-tokenizeしたのちsubword用にtokenizerをtrainする。
'''
import argparse
import logging
import multiprocessing as mp
import os
import pathlib
import subprocess
from typing import Dict, List, Optional, Tuple, Union

import fugashi
from tqdm import tqdm
from tokenizers import normalizers, SentencePieceBPETokenizer, BertWordPieceTokenizer

from utils import get_word_tokenizer

def mp_tokenize(
    input_txt: str,
    output_txt: str,
    num_file: int,
    word_tokenizer_type: str,
    mecab_dic_type: str,
    mecab_option: str,
    sudachi_split_mode: Optional[str],
    sudachi_config_path: Optional[str],
    sudachi_resource_dir: Optional[str],
    sudachi_dict_type: Optional[str]
) -> None:

    main_tokenizer = get_word_tokenizer(
        word_tokenizer_type=word_tokenizer_type,
        do_lower_case=False,
        mecab_dic=mecab_dic_type,
        mecab_option=mecab_option,
        sudachi_split_mode=sudachi_split_mode,
        sudachi_config_path=sudachi_config_path,
        sudachi_resource_dir=sudachi_resource_dir,
        sudachi_dict_type=sudachi_dict_type
    )
    if num_file == 0:
        line_all = int(subprocess.run(['wc', '-l', input_txt], encoding='utf-8', stdout=subprocess.PIPE).stdout.split()[0])
        pbar = tqdm(total=line_all)
    with open(input_txt, 'r') as infile, open(output_txt, 'w') as outfile:
        for line in infile:
            if line == '\n':
                outfile.write('\n')
            else:
                # outfile.write(tagger.parse(line.strip()) + '\n')
                outfile.write(" ".join(main_tokenizer.tokenize(line.strip())) + '\n')
            if num_file == 0:
                pbar.update(1)
    if num_file == 0:
        pbar.close()


def train_tokenizer(
    input_file_or_dir:str,
    output_dir:str,
    vocab_size:int,
    min_frequency:int,
    limit_alphabet:int,
    num_unused_tokens:int,
    tokenizer_type:str,
    language:str
) -> None:

    if os.path.isfile(input_file_or_dir):
        files = [input_file_or_dir]
    elif os.path.isdir(input_file_or_dir):
        files = list(map(str, pathlib.Path(input_file_or_dir).glob('*.txt')))
    else:
        raise ValueError('argument input_file_or_dir must be text file or directory which consists .txt files.')
    
    logger.info('Train tokenizer...')
    os.makedirs(output_dir, exist_ok=True)
    if tokenizer_type=='sentencepiece':
        special_tokens = ['<unused{}>'.format(i) for i in range(args.num_unused_tokens)]
        import sentencepiece as spm
        spm.SentencePieceTrainer.Train(
            input=files,
            # model_dir=output_dir,
            vocab_size=vocab_size,
            model_prefix=os.path.join(output_dir, 'spiece'),
            character_coverage=0.9995,
            num_threads=os.cpu_count(),
            train_extremely_large_corpus=True,
            unk_piece="[UNK]",
            bos_piece="[CLS]",
            eos_piece="[SEP]",
            pad_piece="[PAD]",
            user_defined_symbols=','.join(special_tokens)
        )
    elif tokenizer_type=='wordpiece':
        if language == 'ja':
            tokenizer = BertWordPieceTokenizer(
                handle_chinese_chars=False,
                strip_accents=False,
                lowercase=False
            )
        elif language == 'en':
                tokenizer = BertWordPieceTokenizer(
                handle_chinese_chars=True,
                strip_accents=None, # determined by the value for lowercase
                lowercase=True
            )
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        special_tokens += ['<unused{}>'.format(i) for i in range(args.num_unused_tokens)]
        tokenizer.train(
            files = files, 
            vocab_size = vocab_size,
            min_frequency = min_frequency,
            limit_alphabet = limit_alphabet,
            special_tokens = special_tokens
        )
        # save tokenizer
        tokenizer.save_model(output_dir)
    else:
        raise ValueError(f'Invalid tokenizer_type {tokenizer_type}.')
    logger.info('Tokenizer saved.')

    
if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_tokenizer', required=True, type=str, choices=['mecab', 'juman', 'sudachi', 'spacy-luw', 'none', 'basic'])
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--language', type=str, default='ja', choices=['ja', 'en'])
    # parallel option
    parser.add_argument('--intermediate_dir', type=str, default='tmp')
    parser.add_argument('--num_files', type=int, default=1)
    # pre-tokenize(mainword) option
    parser.add_argument('--pretokenized_prefix', type=str, default='_pretokenized')
    parser.add_argument('--disable_normalize_text', action='store_false')
    # subword training option
    parser.add_argument('--tokenizer_type', required=True, type=str, choices=['sentencepiece', 'wordpiece'])
    parser.add_argument('--vocab_size', type=int, default=32768)
    parser.add_argument('--min_frequency', type=int, default=2, help='only wordpiece')
    parser.add_argument('--limit_alphabet', type=int, default=2900, help='only wordpiece')
    parser.add_argument('--num_unused_tokens', type=int, default=10)
    # mecab option
    parser.add_argument('--mecab_dic_type', type=str, default='', choices=['', 'unidic_lite', 'unidic', 'ipadic'])
    parser.add_argument('--mecab_option', type=str, default='')
    # sudachi option
    parser.add_argument('--sudachi_split_mode', default='C', choices=['A', 'B', 'C'])
    parser.add_argument('--sudachi_config_path')
    parser.add_argument('--sudachi_resource_dir')
    parser.add_argument('--sudachi_dict_type')
    args = parser.parse_args()

    # assertion
    if args.language == 'ja':
        assert args.word_tokenizer != 'basic'
    elif args.language == 'en':
        assert args.word_tokenizer in ['none', 'basic']
    else:
        raise ValueError('Invalid argument language')

    if '.txt' not in args.input_file:
        raise ValueError('input_file must be a txt file')
    if args.num_files < 1:
        raise ValueError('argument num_files must be 1 or larger')

    # logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt = '%(asctime)s %(levelname)s: %(message)s', 
        datefmt = '%Y/%m/%d %H:%M:%S'
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if args.language == 'ja':
        # split
        if args.num_files > 1:
            logger.info('Splitting...')
            line_all = int(subprocess.run(['wc', '-l', args.input_file], encoding='utf-8', stdout=subprocess.PIPE).stdout.split()[0])
            intermediate_plib_dir = pathlib.Path(args.intermediate_dir)
            line_per_file = line_all // args.num_files
            # pbar = tqdm(total=args.num_files)
            cnt_file, cnt_line = 0, 0
            os.makedirs(args.intermediate_dir, exist_ok=True)
            with open(args.input_file, 'r') as infile:
                f = open((intermediate_plib_dir / f'{cnt_file}.txt').resolve(), 'w')
                for line in infile:
                    if cnt_line >= line_per_file and line == '\n':
                        f.write('\n')
                        f.close()
                        # pbar.update(1)
                        cnt_file += 1
                        f = open((intermediate_plib_dir / f'{cnt_file}.txt').resolve(), 'w')
                        cnt_line = 0
                    else:
                        f.write(line)
                        cnt_line += 1
                f.close()
                # pbar.update(1)
                # pbar.close()
            
        # pre-tokenize
        logger.info('Pre-tokenizing...')
        if args.num_files == 1:
            input_plib_file = pathlib.Path(args.input_file)
            pretokenized_plib_file = input_plib_file.parent.joinpath(input_plib_file.stem + args.pretokenized_prefix + '.txt')
            mp_tokenize(
                input_txt=str(input_plib_file),
                output_txt=str(pretokenized_plib_file),
                num_file=0,
                word_tokenizer_type=args.word_tokenizer,
                mecab_dic_type=args.mecab_dic_type,
                mecab_option=args.mecab_option,
                sudachi_split_mode=args.sudachi_split_mode,
                sudachi_config_path=args.sudachi_config_path,
                sudachi_resource_dir=args.sudachi_resource_dir,
                sudachi_dict_type=args.sudachi_dict_type
            )
            logger.info(f'Pre-tokenized files are saved in {str(pretokenized_plib_file)}')
            input_file_or_dir = str(pretokenized_plib_file)
        else:
            pretokenized_plib_dir = intermediate_plib_dir.parent.joinpath(intermediate_plib_dir.stem + args.pretokenized_prefix)
            os.makedirs(pretokenized_plib_dir, exist_ok=True)
            with mp.Pool(args.num_files) as pool:
                mp_task = [pool.apply_async(
                    mp_tokenize, (
                        str((intermediate_plib_dir / f'{i}.txt').resolve()),
                        str((pretokenized_plib_dir / f'{i}.txt').resolve()),
                        i,
                        args.word_tokenizer,
                        args.mecab_dic_type,
                        args.mecab_option,
                        args.sudachi_split_mode,
                        args.sudachi_config_path,
                        args.sudachi_resource_dir,
                        args.sudachi_dict_type
                    )
                ) for i in range(cnt_file+1)]
                _ = [f.get() for f in mp_task]
            logger.info(f'Pre-tokenized files are saved in {str(pretokenized_plib_dir)}')
            input_file_or_dir = str(pretokenized_plib_dir)
    else:
        input_file_or_dir = args.input_file       

    # train tokenizer
    train_tokenizer(
        input_file_or_dir = input_file_or_dir,
        output_dir = args.model_dir,
        vocab_size = args.vocab_size,
        min_frequency = args.min_frequency,
        limit_alphabet = args.limit_alphabet,
        num_unused_tokens = args.num_unused_tokens,
        tokenizer_type = args.tokenizer_type,
        language = args.language
    )
