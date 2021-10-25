import argparse
import datetime as dt
from fractions import Fraction
import json
import os
import warnings

import datasets
import torch
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, normalizers
from tokenizers.processors import BertProcessing
import transformers
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import (
    BertConfig,
    BertForPreTraining,
    ElectraConfig,
    logging,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments
)
logging.set_verbosity_info()
logging.enable_explicit_format()

import utils
TrainingArguments._setup_devices = utils._setup_devices

warnings.simplefilter('ignore', UserWarning)
assert utils.TorchVersion(torch.__version__) >= utils.TorchVersion('1.8.0'), f'This code requires a minimum version of PyTorch of 1.8.0, but the version found is {torch.__version__}'
transformers.utils.check_min_version('4.7.0')

logger = transformers.logging.get_logger()


def get_model_bert(
        tokenizer:PreTrainedTokenizerBase,
        param_config:dict,
    ) -> PreTrainedModel:
    
    bert_config = BertConfig(
        vocab_size = tokenizer.vocab_size, 
        hidden_size = param_config['hidden-size'], 
        num_hidden_layers = param_config['number-of-layers'],
        num_attention_heads = param_config['attention-heads'],
        intermediate_size = param_config['ffn-inner-hidden-size'],
        max_position_embeddings = param_config['sequence-length'],
    )
    model = BertForPreTraining(config=bert_config)
    return model


def get_model_electra(
        tokenizer:PreTrainedTokenizerBase,
        param_config:dict,
    ) -> PreTrainedModel:

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
    return model


def run_pretraining(
        tokenizer:PreTrainedTokenizerBase, 
        dataset_dir:str,
        model_name:str,
        model_dir:str,
        param_config:dict,
        fp16_type:int,
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
    training_args = TrainingArguments(
        output_dir = model_dir,
        do_train = True,
        do_eval = False, # default
        per_device_train_batch_size = per_device_train_batch_size,
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
        fp16 = bool(fp16_type!=0),
        fp16_opt_level = f"O{fp16_type}", 
        #:"O1":Mixed Precision (recommended for typical use), "O2":“Almost FP16” Mixed Precision, "O3":FP16 training
        disable_tqdm = True,
        max_steps = param_config['train-steps'],
        gradient_accumulation_steps = 1 if 'accumulation-steps' not in param_config.keys() else param_config['accumulation-steps'],
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

    # dataset
    dataset = datasets.load_from_disk(dataset_dir)
    dataset.set_format(type='torch')
    logger.info('Dataset is loaded')

    # model
    if model_name == 'bert':
        model = get_model_bert(tokenizer, param_config)
    elif model_name == 'electra':
        model = get_model_electra(tokenizer, param_config)
    logger.info(f'{model_name} mode is loaded')    

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
    trainoutput = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
        do_log_loss_gen_disc=bool(model_name == 'electra')
    )


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True, 
                        help="uploaded name in HuggingFace Hub or directory path containing vocab.txt")
    parser.add_argument("--dataset_dir", type=str, required=True, help="directory of corpus dataset")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--parameter_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    # optional
    parser.add_argument('--fp16_type', type=int, default=0, choices=[0,1,2,3], 
                        help='default:0(disable), see https://nvidia.github.io/apex/amp.html for detail')
    
    parser.add_argument('--tokenizer_type', type=str, default='', choices=['', 'sentencepiece', 'wordpiece'])
    parser.add_argument('--mecab_dic_type', type=str, default='', choices=['', 'unidic_lite', 'unidic', 'ipadic'])
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--do_whole_word_mask', action='store_true')
    parser.add_argument('--do_continue', action='store_true')
    parser.add_argument('--node_rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    assert not (args.tokenizer_type=='sentencepiece' and args.do_whole_word_mask), 'Whole Word Masking cannot be applied with sentencepiece tokenizer'

    # global variables
    datasets.config.IN_MEMORY_MAX_SIZE = 120 * 10**9

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
    
    tokenizer = utils.load_tokenizer(
        tokenizer_name_or_path = args.tokenizer_name_or_path,
        tokenizer_type = args.tokenizer_type,
        mecab_dic_type = args.mecab_dic_type,
    )
    run_pretraining(
        tokenizer = tokenizer,
        dataset_dir = args.dataset_dir,
        model_name = model_name,
        model_dir = args.model_dir,
        param_config = param_config,
        fp16_type = args.fp16_type,
        do_whole_word_mask = args.do_whole_word_mask,
        do_continue = args.do_continue,
        node_rank = args.node_rank,
        local_rank = args.local_rank,
        run_name = args.run_name,
    )