'''
Pre-tokenize用にファイルを分割する。
MeCabでPre-tokenizeしたのちsubword用にtokenizerをtrainする。
'''
import argparse
import logging
import os
import pathlib
import subprocess
from typing import Dict, List, Optional, Tuple, Union

import fugashi
from tqdm import tqdm
from tokenizers import normalizers, SentencePieceBPETokenizer, BertWordPieceTokenizer

def mp_tokenize(
        input_txt:str,
        output_txt:str,
        mecab_dic_type: str,
        mecab_option: str,
        num_file:int
    ) -> None:

    option = '-Owakati '
    if mecab_dic_type != '':
        if mecab_dic_type == "unidic_lite":
            import unidic_lite
            option += f'-d {unidic_lite.DICDIR} '
        elif mecab_dic_type == "unidic":
            import unidic
            option += f'-d {unidic.DICDIR} '
        elif mecab_dic_type == "ipadic":
            import ipadic
            option += f'-d {ipadic.DICDIR} '
        else:
            raise ValueError("Invalid mecab_dic_type is specified.")

    if mecab_option != '':
        option += f'-r {mecab_option}'

    tagger = fugashi.GenericTagger(option)
    if num_file == 0:
        line_all = int(subprocess.run(['wc', '-l', input_txt], encoding='utf-8', stdout=subprocess.PIPE).stdout.split()[0])
        pbar = tqdm(total=line_all)
    with open(input_txt, 'r') as infile, open(output_txt, 'w') as outfile:
        for line in infile:
            if line == '\n':
                outfile.write('\n')
            else:
                outfile.write(tagger.parse(line.strip()) + '\n')
            if num_file == 0:
                pbar.update(1)
    if num_file == 0:
        pbar.close()


def train_tokenizer(
        input_file_or_dir:str,
        output_dir:str,
        vocab_size:int,
        min_frequency:int,
        tokenizer_type:str,
    ) -> None:

    if os.path.isfile(input_file_or_dir):
        files = [input_file_or_dir]
    elif os.path.isdir(input_file_or_dir):
        files = list(map(str, pathlib.Path(input_file_or_dir).glob('*.txt')))
    else:
        raise ValueError('argument input_file_or_dir must be text file or directory which consists .txt files.')
    
    # Initialize a tokenizer
    if tokenizer_type=='sentencepiece':
        tokenizer = SentencePieceBPETokenizer(
            unk_token="[UNK]", 
            add_prefix_space=False, # 文頭に自動でスペースを追加しない
        )
        # 改行がinput_fileにあるとtokenも改行がついてくるのでstrip
        # cf. https://github.com/huggingface/tokenizers/issues/231
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.Strip(),
            normalizers.NFKC(),
            # normalizers.BertNormalizer(
            #     handle_chinese_chars = False,
            #     lowercase = True,
            # )
        ])
    elif tokenizer_type=='wordpiece':
        tokenizer = BertWordPieceTokenizer(handle_chinese_chars=False)
    else:
        raise ValueError(f'Invalid tokenizer_type {tokenizer_type}.')    

    logger.info('Train tokenizer...')
    tokenizer.train(
        files = files, 
        vocab_size = vocab_size, 
        min_frequency = min_frequency, 
        special_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
        ]
    )

    # save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    logger.info('Tokenizer saved.')

    
if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--model_dir', required=True, type=str)
    # split option
    parser.add_argument('--intermediate_dir', type=str, default='')
    parser.add_argument('--num_files', type=int, default=1)
    # pre-tokenize option
    parser.add_argument('--pretokenized_prefix', type=str, default='_pretokenized')
    parser.add_argument('--mecab_dic_type', type=str, default='', choices=['', 'unidic_lite', 'unidic', 'ipadic'])
    parser.add_argument('--mecab_option', type=str, default='')
    # train tokenize option
    parser.add_argument('--tokenizer_type', required=True, type=str, choices=['sentencepiece', 'wordpiece'])
    parser.add_argument('--vocab_size', type=int, default=30522)
    parser.add_argument('--min_frequency', type=int, default=2)
    args = parser.parse_args()
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
            pbar.update(1)
            pbar.close()
        
    # pre-tokenize
    logger.info('Pre-tokenizing...')
    if args.num_files == 1:
        input_plib_file = pathlib.Path(args.input_file)
        pretokenized_plib_file = input_plib_file.parent.joinpath(input_plib_file.stem + args.pretokenized_prefix + '.txt')
        mp_tokenize(str(input_plib_file), str(pretokenized_plib_file), args.mecab_dic_type, args.mecab_option, 0)
        logger.info(f'Pre-tokenized files are saved in {str(pretokenized_plib_file)}')
    else:
        pretokenized_plib_dir = intermediate_plib_dir.parent.joinpath(intermediate_plib_dir.stem + args.pretokenized_prefix)
        os.makedirs(pretokenized_plib_dir, exist_ok=True)
        with mp.Pool(args.num_files) as pool:
            mp_task = [pool.apply_async(mp_tokenize, (
                str((intermediate_plib_dir / f'{i}.txt').resolve()),
                str((pretokenized_plib_dir / f'{i}.txt').resolve()),
                args.mecab_dic_type, args.mecab_option, i
            )) for i in range(cnt_file+1)]
            _ = [f.get() for f in mp_task]
        logger.info(f'Pre-tokenized files are saved in {str(pretokenized_plib_dir)}')
       

    # train tokenizer
    input_file_or_dir = str(pretokenize_plib_dir) if args.num_files > 1 else str(pretokenized_plib_file)
    train_tokenizer(
        input_file_or_dir = input_file_or_dir,
        output_dir = args.output_dir,
        vocab_size = args.vocab_size,
        min_frequency = args.min_frequency,
        tokenizer_type = args.tokenizer_type,
    )
