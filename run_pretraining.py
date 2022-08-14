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
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    ElectraConfig,
    logging,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    RobertaConfig,
    RobertaForMaskedLM,
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
        load_pretrained:bool,
        param_config:dict,
    ) -> PreTrainedModel:
    
    if load_pretrained:
        model = BertForPreTraining.from_pretrained(param_config['pretrained_model_name_or_path'])
        flozen_layers = param_config['flozen-layers']
        if flozen_layers > -1:
            for name, param in model.bert.embeddings.named_parameters():
                param.requires_grad = False
            for i in range(flozen_layers):
                for name, param in model.bert.encoder.layer[i].named_parameters():
                    param.requires_grad = False
    else:
        bert_config = BertConfig(
            pad_token_id = tokenizer.pad_token_id,
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
        load_pretrained:bool,
        param_config:dict,
    ) -> PreTrainedModel:

    if load_pretrained:
        model = utils.ElectraForPretrainingModel.from_pretrained(
            param_config['pretrained_generator_model_name_or_path'],
            param_config['pretrained_discriminator_model_name_or_path']
        )
        flozen_layers = param_config['flozen-layers']
        if flozen_layers > -1:
            for m in [model.generator, model.discriminator]:
                for name, param in m.electra.embeddings.named_parameters():
                    param.requires_grad = False
                for i in range(flozen_layers):
                    for name, param in m.electra.encoder.layer[i].named_parameters():
                        param.requires_grad = False
    else:
        frac_generator = Fraction(param_config['generator-size'])
        config_generator = ElectraConfig(
            pad_token_id = tokenizer.pad_token_id,
            vocab_size = tokenizer.vocab_size, 
            embedding_size = param_config['embedding-size'],
            hidden_size = int(param_config['hidden-size'] * frac_generator), 
            num_attention_heads = int(param_config['attention-heads'] * frac_generator),
            num_hidden_layers = param_config['number-of-layers'],
            intermediate_size = int(param_config['ffn-inner-hidden-size'] * frac_generator),
            max_position_embeddings = param_config['sequence-length'],
        )
        config_discriminator = ElectraConfig(
            pad_token_id = tokenizer.pad_token_id,
            vocab_size = tokenizer.vocab_size, 
            embedding_size = param_config['embedding-size'],
            hidden_size = param_config['hidden-size'], 
            num_attention_heads = param_config['attention-heads'],
            num_hidden_layers = param_config['number-of-layers'],
            intermediate_size = param_config['ffn-inner-hidden-size'],
            max_position_embeddings = param_config['sequence-length'],
        )
        model = utils.ElectraForPretrainingModel(
            config_generator = config_generator,
            config_discriminator = config_discriminator,
        )
    return model


def get_model_roberta(
        tokenizer:PreTrainedTokenizerBase,
        load_pretrained:bool,
        param_config:dict,
    ) -> PreTrainedModel:
    
    if load_pretrained:
        model = RobertaForMaskedLM.from_pretrained(param_config['pretrained_model_name_or_path'])
        flozen_layers = param_config['flozen-layers']
        if flozen_layers > -1:
            for name, param in model.roberta.embeddings.named_parameters():
                param.requires_grad = False
            for i in range(flozen_layers):
                for name, param in model.roberta.encoder.layer[i].named_parameters():
                    param.requires_grad = False
    else:
        roberta_config = RobertaConfig(
            pad_token_id = tokenizer.pad_token_id,
            bos_token_id = tokenizer.cls_token_id,
            eos_token_id = tokenizer.sep_token_id,
            vocab_size = tokenizer.vocab_size, 
            hidden_size = param_config['hidden-size'], 
            num_hidden_layers = param_config['number-of-layers'],
            num_attention_heads = param_config['attention-heads'],
            intermediate_size = param_config['ffn-inner-hidden-size'],
            max_position_embeddings = param_config['sequence-length'] + tokenizer.pad_token_id + 1,
        )
        model = RobertaForMaskedLM(config=roberta_config)
    return model


def get_model_deberta(
        tokenizer:PreTrainedTokenizerBase,
        load_pretrained:bool,
        param_config:dict,
    ) -> PreTrainedModel:
    
    if load_pretrained:
        model = DebertaV2ForMaskedLM.from_pretrained(param_config['pretrained_model_name_or_path'])
        flozen_layers = param_config['flozen-layers']
        if flozen_layers > -1:
            for name, param in model.deberta.embeddings.named_parameters():
                param.requires_grad = False
            for i in range(flozen_layers):
                for name, param in model.deberta.encoder.layer[i].named_parameters():
                    param.requires_grad = False
    else:
        deberta_config = DebertaV2Config(
            pad_token_id = tokenizer.pad_token_id,
            bos_token_id = tokenizer.cls_token_id,
            eos_token_id = tokenizer.sep_token_id,
            vocab_size = tokenizer.vocab_size, 
            hidden_size = param_config['hidden-size'], 
            num_hidden_layers = param_config['number-of-layers'],
            num_attention_heads = param_config['attention-heads'],
            intermediate_size = param_config['ffn-inner-hidden-size'],
            max_position_embeddings = param_config['sequence-length'],
        )
        model = DebertaV2ForMaskedLM(config=deberta_config)
    return model


def run_pretraining(
        tokenizer:PreTrainedTokenizerBase, 
        dataset_dir:str,
        model_name:str,
        model_dir:str,
        load_pretrained:bool,
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
    # if do_continue:
    #     training_args = torch.load(os.path.join(model_dir, "training_args.bin"))
    #     per_device_train_batch_size = training_args.per_device_train_batch_size
    # else:
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
        model = get_model_bert(tokenizer, load_pretrained, param_config)
    elif model_name == 'electra':
        model = get_model_electra(tokenizer, load_pretrained, param_config)
    elif model_name == 'roberta':
        model = get_model_roberta(tokenizer, load_pretrained, param_config)
    elif model_name == 'deberta':
        model = get_model_deberta(tokenizer, load_pretrained, param_config)
    logger.info(f'{model_name} model is loaded')    

    # data collator
    if model_name in ['bert', 'roberta', 'deberta']:
        mlm_probability = 0.15
    elif model_name == 'electra':
        mlm_probability = param_config['mask-percent']/100
    if do_whole_word_mask:
        if model_name in ['bert', 'roberta', 'deberta']:
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
        if model_name in ['bert', 'roberta', 'deberta']:
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
    # TODO: implement deepspeed
    trainer = utils.MyTrainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset,
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


def assert_config(param_config:dict, model_type:str, local_rank:int) -> bool:
    """
    Return
    model_name: str, bert, deberta, electra, or roberta
    load_pretrained: bool, True when further pretrain and False when pretrain from scratch
    """
    if model_type not in param_config:
        raise KeyError(f'{model_type} not in parameters.json')
    if 'electra-' in model_type.lower():
        model_name = 'electra'
    elif 'roberta-' in model_type.lower():
        model_name = 'roberta'
    elif 'deberta-' in model_type.lower():
        model_name = 'deberta'
    elif 'bert-' in model_type.lower():
        model_name = 'bert'
    else:
        raise ValueError('Argument model_type must contain electra or bert')
    param_config = param_config[model_type]
    if len(param_config.keys() & {f'pretrained_{x}model_name_or_path' for x in ['', 'generator_', 'discriminator']}) > 0:
        load_pretrained = True
        set_assert = {'flozen-layers'}
        if model_name in ['bert', 'roberta', 'deberta']:
            set_assert = set_assert | {'pretrained_model_name_or_path'}
        if model_name == 'electra':
            set_assert = set_assert | {f'pretrained_{x}_model_name_or_path' for x in ['generator', 'discriminator']}
    else:
        load_pretrained = False
        set_assert = {
            'number-of-layers', 'hidden-size', 'sequence-length', 'ffn-inner-hidden-size', 'attention-heads',
            'warmup-steps', 'learning-rate', 'batch-size', 'train-steps'
        }
        if model_name == 'electra':
            set_assert = set_assert | {'embedding-size', 'generator-size', 'mask-percent'}
    if param_config.keys() < set_assert:
        raise ValueError(f'{set_assert-param_config.keys()} is(are) not in parameter_file')
    if str(local_rank) not in param_config['batch-size'].keys():
        raise ValueError(f'local_rank {local_rank} is not defined in batch-size of parameter_file')
    logger.info(f'Config[{model_type}] is loaded')
    return model_name, load_pretrained
    


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # required
    parser.add_argument("--dataset_dir", type=str, required=True, help="directory of corpus dataset")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--parameter_file', type=str, required=True, help="json file defining model parameters")
    parser.add_argument('--model_type', type=str, required=True, help="model parameter defined in the parameter_file. It must contain 'bert-', 'deberta-', 'electra-', or 'roberta-'")
    # optional
    parser.add_argument('--fp16_type', type=int, default=0, choices=[0,1,2,3], 
                        help='default:0(disable), see https://nvidia.github.io/apex/amp.html for detail')
    
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--do_whole_word_mask', action='store_true')
    parser.add_argument('--do_continue', action='store_true')
    parser.add_argument('--node_rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    utils.add_arguments_for_tokenizer(parser)
    args = parser.parse_args()
    assert not (args.tokenizer_type=='sentencepiece' and args.do_whole_word_mask), 'Whole Word Masking cannot be applied with sentencepiece tokenizer'
    utils.assert_arguments_for_tokenizer(args)

    # global variables
    datasets.config.IN_MEMORY_MAX_SIZE = 50 * 10**9

    # parameter configuration
    with open(args.parameter_file, 'r') as f:
        param_config = json.load(f)
    model_name, load_pretrained = assert_config(param_config, args.model_type, args.local_rank)
    
    tokenizer = utils.load_tokenizer(args)

    run_pretraining(
        tokenizer = tokenizer,
        dataset_dir = args.dataset_dir,
        model_name = model_name,
        model_dir = args.model_dir,
        load_pretrained = load_pretrained,
        param_config = param_config[args.model_type],
        fp16_type = args.fp16_type,
        do_whole_word_mask = args.do_whole_word_mask,
        do_continue = args.do_continue,
        node_rank = args.node_rank,
        local_rank = args.local_rank,
        run_name = args.run_name,
    )
