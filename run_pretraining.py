import argparse
import datetime as dt
from fractions import Fraction
import json
import logging
import os
from typing import Tuple
import warnings

import torch
from torch.utils.data.dataset import Dataset
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, normalizers
from tokenizers.processors import BertProcessing
import transformers
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import (
    BertConfig,
    BertForPreTraining,
    ElectraConfig,
    PreTrainedModel,
)
transformers.logging.set_verbosity_info()
transformers.logging.enable_explicit_format()

import utils


warnings.simplefilter('ignore', UserWarning)
assert any(v in torch.__version__ for v in ['1.9.0']), f'This file is only guranteed with pytorch 1.9.0, but this is {torch.__version__}'
assert transformers.__version__ in ['4.7.0'], f'This file is only guranteed with transformers 4.7.0, but this is {transformers.__version__}'


def load_tokenizer(tokenizer_dir:str, max_length:int,
                   tokenizer_type:str,
                   mecab_dic_type:str,
                   ) -> transformers.tokenization_utils_base.PreTrainedTokenizerBase:
    # load tokenizer
    if tokenizer_type=='sentencepiece':
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
        tokenizer.enable_truncation(max_length = max_length)
        # convert to transformers style
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object = tokenizer,
            model_max_length = max_length,
            unk_token = "[UNK]",
            sep_token = "[SEP]",
            pad_token = "[PAD]",
            cls_token = "[CLS]",
            mask_token = "[MASK]",
        )
    elif tokenizer_type=='wordpiece':
        tokenizer = transformers.BertJapaneseTokenizer(
            os.path.join(tokenizer_dir, "vocab.txt"),
            do_lower_case = False,
            word_tokenizer_type = "mecab",
            subword_tokenizer_type = "wordpiece",
            tokenize_chinese_chars = False,
            mecab_kwargs = {'mecab_dic': mecab_dic_type},
            model_max_length = max_length
        )
    else:
        raise ValueError(f'Invalid tokenizer_type {tokenizer_type}.')

    return tokenizer


def make_dataset_model_bert(
        tokenizer:transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        input_file:str,
        param_config:dict,
    ) -> Tuple[Dataset, PreTrainedModel]:

    dataset = utils.TextDatasetForNextSentencePrediction(
        tokenizer = tokenizer, 
        file_path = input_file, 
        overwrite_cache= False,
        block_size = param_config['sequence-length'],
        short_seq_probability = 0.1, # default
        nsp_probability = 0.5, # default
    )
    bert_config = BertConfig(
        vocab_size = tokenizer.vocab_size, 
        hidden_size = param_config['hidden-size'], 
        num_hidden_layers = param_config['number-of-layers'],
        num_attention_heads = param_config['attention-heads'],
        intermediate_size = param_config['ffn-inner-hidden-size'],
        max_position_embeddings = param_config['sequence-length'],
    )
    model = BertForPreTraining(config=bert_config)

    return dataset, model


def make_dataset_model_electra(
        tokenizer:transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        input_file:str,
        param_config:dict,
    ) -> Tuple[Dataset, PreTrainedModel]:

    dataset = utils.LineByLineTextDataset(
        tokenizer = tokenizer, 
        file_path = input_file, 
        overwrite_cache = False,
        block_size = param_config['sequence-length'],
    )
    frac_generator = Fraction(param_config['generator-size'])
    config_generator = ElectraConfig(
        vocab_size = tokenizer.vocab_size, 
        embedding_size = param_config['embedding-size'],
        hidden_size = int(param_config['hidden-size'] * frac_generator), 
        num_attention_heads = int(param_config['attention-heads'] * frac_generator),
        num_hidden_layers = param_config['number-of-layers'],
        intermediate_size = int(param_config['ffn-inner-hidden-size'] * frac_generator),
    )
    config_discriminator = ElectraConfig(
        vocab_size = tokenizer.vocab_size, 
        embedding_size = param_config['embedding-size'],
        hidden_size = param_config['hidden-size'], 
        num_attention_heads = param_config['attention-heads'],
        num_hidden_layers = param_config['number-of-layers'],
        intermediate_size = param_config['ffn-inner-hidden-size'],
    )
    model = utils.ElectraForPretrainingModel(
        config_generator = config_generator,
        config_discriminator = config_discriminator,
    )

    return dataset, model


def run_pretraining(
        tokenizer:transformers.tokenization_utils_base.PreTrainedTokenizerBase, 
        input_file:str,
        model_name:str,
        model_dir:str,
        param_config:dict,
        do_whole_word_mask:bool,
        do_continue:bool,
        node_rank:int,
        local_rank:int,
        run_name:str
    ) -> None:
    if run_name == '':
        run_name = dt.datetime.now().strftime("%y%m%d") + "_" + os.path.basename(os.path.dirname(model_dir))
    os.makedirs(model_dir, exist_ok=True)

    # training argument
    if do_continue:
        training_args = torch.load(os.path.join(model_dir, "training_args.bin"))
        per_device_train_batch_size = training_args.per_device_train_batch_size
    else:
        if torch.cuda.device_count() > 0:
            per_device_train_batch_size = int(param_config['batch-size'][str(node_rank)] / torch.cuda.device_count())
        else:
            per_device_train_batch_size = param_config['batch-size'][str(node_rank)]
    # initialize
    training_args = transformers.TrainingArguments(
        output_dir = model_dir,
        do_train = True,
        do_eval = False, # default
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = 1, # default
        learning_rate = param_config['learning-rate'], 
        adam_beta1 = 0.9, # same as BERT paper
        adam_beta2 = 0.999, # same as BERT paper
        adam_epsilon = 1e-6,
        weight_decay = 0.01, # same as BERT paper
        warmup_steps = param_config['warmup-steps'], 
        logging_dir = os.path.join(os.path.dirname(__file__), f"runs/{run_name}"),
        save_steps = param_config['save-steps'] if 'save-steps' in param_config.keys() else 50000, #default:500
        save_strategy = "steps", # default:"steps"
        logging_steps = param_config['logging-steps'] if 'logging-steps' in param_config.keys() else 5000, # default:500
        save_total_limit = 20, # optional
        seed = 42, # default
        fp16 = bool(torch.cuda.device_count()>0),
        fp16_opt_level = "O2", #:Mixed Precision (recommended for typical use), "O2":“Almost FP16” Mixed Precision, "O3":FP16 training
        disable_tqdm = True,
        max_steps = param_config['train-steps'],
        dataloader_num_workers = 3,
        dataloader_pin_memory=False,
        local_rank = local_rank,
        report_to = "tensorboard"
    )
    if not do_continue:
        if local_rank != -1:
            if torch.cuda.device_count() > 0:
                training_args.per_device_train_batch_size = int(param_config['batch-size'][str(node_rank)] / torch.cuda.device_count())
            else:
                training_args.per_device_train_batch_size = param_config['batch-size'][str(node_rank)]
        torch.save(training_args, os.path.join(model_dir, "training_args.bin"))

    # dataset and model
    if model_name == 'bert':
        train_dataset, model = make_dataset_model_bert(tokenizer, input_file, param_config)
    elif model_name == 'electra':
        train_dataset, model = make_dataset_model_electra(tokenizer, input_file, param_config)
    logger.info('Dataset was complete.')

    # data collator
    if model_name == 'bert':
        mlm_probability = 0.15
    elif model_name == 'electra':
        mlm_probability = param_config['mask-percent']/100
    if do_whole_word_mask:
        if model_name == 'bert':
            data_collator = utils.DataCollatorForWholeWordMask(
                tokenizer = tokenizer, 
                mlm = True,
                mlm_probability = mlm_probability
            )
        elif model_name == 'electra':
            # https://github.com/google-research/electra/issues/57
            data_collator = utils.DataCollatorForWholeWordMask(
                tokenizer = tokenizer, 
                mlm = True,
                mlm_probability = mlm_probability,
                rate_replaced = 0.85,
                rate_random = 0,
                rate_unchanged = 0.15
            )
    else:
        if model_name == 'bert':
            data_collator = DataCollatorForLanguageModeling(
                tokenizer = tokenizer,
                mlm = True,
                mlm_probability = mlm_probability
            )
        elif model_name == 'electra':
            data_collator = utils.DataCollatorForLanguageModelingWithElectra(
                tokenizer = tokenizer,
                mlm = True,
                mlm_probability = mlm_probability
            )
    logger.info('Datacollator was complete.')
    
    trainer = utils.MyTrainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        node_rank = node_rank
    )
    trainer.batch_config = param_config['batch-size']
    trainer.real_batch_size = sum(param_config['batch-size'].values())

    logger.info('Pretraining starts.')
    resume_from_checkpoint = True if do_continue else None
    trainoutput = trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--parameter_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--tokenizer_type', type=str, choices=['sentencepiece', 'wordpiece'])
    parser.add_argument('--mecab_dic_type', type=str, default='', choices=['', 'unidic_lite', 'unidic', 'ipadic'])
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--do_whole_word_mask', action='store_true')
    parser.add_argument('--do_continue', action='store_true')
    parser.add_argument('--node_rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    assert not (args.tokenizer_type=='sentencepiece' and args.do_whole_word_mask), 'Whole Word Masking cannot be applied with sentencepiece tokenizer'

    # get root logger
    # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # parameter configuration
    with open(args.parameter_file, 'r') as f:
        param_config = json.load(f)
    if args.model_type not in param_config:
        raise KeyError(f'{args.model_type} not in parameters.json')
    if 'electra-' in args.model_type.lower():
        model_name = 'electra'
    elif 'bert-' in args.model_type.lower():
        model_name = 'bert'
    else:
        raise ValueError('Argument model_type must contain electra or bert')
    param_config = param_config[args.model_type]
    set_assert = {
        'number-of-layers', 'hidden-size', 'sequence-length', 'ffn-inner-hidden-size', 'attention-heads',
        'warmup-steps', 'learning-rate', 'batch-size', 'train-steps'
    }
    if model_name == 'electra':
        set_assert = set_assert | set(['embedding-size', 'generator-size', 'mask-percent'])
    if param_config.keys() < set_assert:
        raise ValueError(f'{set_assert-param_config.keys()} is(are) not in parameter_file')
    if str(args.local_rank) not in param_config['batch-size'].keys():
        raise ValueError(f'local_rank {args.local_rank} is not defined in batch-size of parameter_file')
    logger.info(f'Config[{args.model_type}] is loaded')
    
    # tokenizer
    tokenizer = load_tokenizer(
        tokenizer_dir = args.tokenizer_dir,
        tokenizer_type = args.tokenizer_type,
        max_length = param_config['sequence-length'],
        mecab_dic_type = args.mecab_dic_type,
    )
    run_pretraining(
        tokenizer = tokenizer,
        input_file = args.input_file,
        model_name = model_name,
        model_dir = args.model_dir,
        param_config = param_config,
        do_whole_word_mask = args.do_whole_word_mask,
        do_continue = args.do_continue,
        node_rank = args.node_rank,
        local_rank = args.local_rank,
        run_name = args.run_name
    )