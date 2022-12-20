import argparse
import datetime as dt
import json
import logging
import os
import warnings
from fractions import Fraction
from typing import Dict, Union

import datasets
import psutil
import torch
import transformers
from transformers.models import auto
from transformers.data.data_collator import DataCollatorMixin, DataCollatorWithPadding
from transformers import (
    ElectraConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

transformers.logging.set_verbosity_warning()
transformers.logging.enable_explicit_format()

import utils
from utils.logger import make_logger_setting
from models import DebertaEmdForPreTraining, DebertaV2EmdForPreTraining


# logger
logger: logging.Logger = logging.getLogger(__name__)
make_logger_setting(logger)


TrainingArguments._setup_devices = utils._setup_devices

warnings.simplefilter("ignore", UserWarning)
assert utils.TorchVersion(torch.__version__) >= utils.TorchVersion(
    "1.8.0"
), f"This code requires a minimum version of PyTorch of 1.8.0, but the version found is {torch.__version__}"
transformers.utils.check_min_version("4.10.2")


def get_model(
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    load_pretrained: bool,
    param_config: dict,
) -> PreTrainedModel:
    model: PreTrainedModel
    if model_name == "electra":
        model = get_model_electra(
            tokenizer=tokenizer,
            load_pretrained=load_pretrained,
            param_config=param_config,
        )
    else:
        if model_name == "debertav2":
            model_name = "deberta-v2"
        MappedConfig = auto.CONFIG_MAPPING[model_name]
        if model_name == "deberta":
            MappedModel = DebertaEmdForPreTraining
        elif model_name == "deberta-v2":
            MappedModel = DebertaV2EmdForPreTraining
        else:
            MappedModel = auto.MODEL_FOR_PRETRAINING_MAPPING[MappedConfig]
        if load_pretrained:
            model = MappedModel.from_pretrained(
                param_config["pretrained_model_name_or_path"]
            )
            flozen_layers = param_config["flozen-layers"]
            if flozen_layers > -1:
                for name, param in getattr(
                    model, model_name.split("-")[0]
                ).embeddings.named_parameters():
                    param.requires_grad = False
                for i in range(flozen_layers):
                    for name, param in (
                        getattr(model, model_name.split("-")[0])
                        .encoder.layer[i]
                        .named_parameters()
                    ):
                        param.requires_grad = False
        else:
            dct_kwargs: Dict[str, int] = {
                "pad_token_id": tokenizer.pad_token_id,
                "vocab_size": tokenizer.vocab_size,
                "hidden_size": param_config["hidden-size"],
                "num_hidden_layers": param_config["number-of-layers"],
                "num_attention_heads": param_config["attention-heads"],
                "intermediate_size": param_config["ffn-inner-hidden-size"],
            }
            if model_name == "roberta":
                dct_kwargs["bos_token_id"] = tokenizer.cls_token_id
                dct_kwargs["eos_token_id"] = tokenizer.sep_token_id
                dct_kwargs["max_position_embeddings"] = (
                    param_config["sequence-length"] + tokenizer.pad_token_id + 1
                )
                dct_kwargs["layer_norm_eps"] = 1e-5
                dct_kwargs["token_vocab_size"] = 1
            else:
                # for bert, deberta, roberta
                dct_kwargs["max_position_embeddings"] = param_config["sequence-length"]
                if model_name in ["deberta", "deberta-v2"]:
                    dct_kwargs["token_vocab_size"] = 0
                    dct_kwargs["relative_attention"] = True
                    dct_kwargs["position_biased_input"] = False
                    dct_kwargs["pos_att_type"] = "c2p|p2c"
                    dct_kwargs["max_relative_positions"] = -1
            config = MappedConfig(**dct_kwargs)
            model = MappedModel(config=config)
    return model


def get_model_electra(
    tokenizer: PreTrainedTokenizerBase,
    load_pretrained: bool,
    param_config: dict,
) -> PreTrainedModel:

    if load_pretrained:
        model = utils.ElectraForPretrainingModel.from_pretrained(
            param_config["pretrained_generator_model_name_or_path"],
            param_config["pretrained_discriminator_model_name_or_path"],
        )
        flozen_layers = param_config["flozen-layers"]
        if flozen_layers > -1:
            for m in [model.generator, model.discriminator]:
                for name, param in m.electra.embeddings.named_parameters():
                    param.requires_grad = False
                for i in range(flozen_layers):
                    for name, param in m.electra.encoder.layer[i].named_parameters():
                        param.requires_grad = False
    else:
        frac_generator = Fraction(param_config["generator-size"])
        config_generator = ElectraConfig(
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
            embedding_size=param_config["embedding-size"],
            hidden_size=int(param_config["hidden-size"] * frac_generator),
            num_attention_heads=int(param_config["attention-heads"] * frac_generator),
            num_hidden_layers=param_config["number-of-layers"],
            intermediate_size=int(
                param_config["ffn-inner-hidden-size"] * frac_generator
            ),
            max_position_embeddings=param_config["sequence-length"],
        )
        config_discriminator = ElectraConfig(
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
            embedding_size=param_config["embedding-size"],
            hidden_size=param_config["hidden-size"],
            num_attention_heads=param_config["attention-heads"],
            num_hidden_layers=param_config["number-of-layers"],
            intermediate_size=param_config["ffn-inner-hidden-size"],
            max_position_embeddings=param_config["sequence-length"],
        )
        model = utils.ElectraForPretrainingModel(
            config_generator=config_generator,
            config_discriminator=config_discriminator,
        )
    return model


def run_pretraining(
    tokenizer: PreTrainedTokenizerBase,
    dataset_dir: str,
    model_name: str,
    model_dir: str,
    load_pretrained: bool,
    param_config: dict,
    is_dataset_masked: bool,
    do_continue: bool = False,
    do_whole_word_mask: bool = False,
    use_deepspeed: bool = False,
    deepspeed_bucket_size: float = 5e8,
    node_rank: int = -1,
    local_rank: int = -1,
    run_name: str = "",
) -> None:
    assert (
        model_name != "bert" or not is_dataset_masked
    ), "Pre-masking with bert is not available"
    if run_name == "":
        run_name = (
            dt.datetime.now().strftime("%y%m%d")
            + "_"
            + os.path.basename(os.path.dirname(model_dir))
        )
    os.makedirs(model_dir, exist_ok=True)

    # training argument
    # if do_continue:
    #     training_args = torch.load(os.path.join(model_dir, "training_args.bin"))
    #     per_device_train_batch_size = training_args.per_device_train_batch_size
    # else:
    if torch.cuda.device_count() > 0:
        per_device_train_batch_size = int(
            param_config["batch-size"][str(node_rank)] / torch.cuda.device_count()
        )
    else:
        per_device_train_batch_size = param_config["batch-size"][str(node_rank)]
    logging_steps = (
        param_config["logging-steps"]
        if "logging-steps" in param_config.keys()
        else 5000
    )
    gradient_accumulation_steps = (
        1
        if "accumulation-steps" not in param_config.keys()
        else param_config["accumulation-steps"]
    )
    if use_deepspeed:
        deepspeed = {
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto",
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto",
                    "warmup_type": "linear"
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "allgather_partitions": True,
                "allgather_bucket_size": deepspeed_bucket_size,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": deepspeed_bucket_size,
                "contiguous_gradients": True,
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "steps_per_print": 999999999999, #logging_steps * gradient_accumulation_steps,
            # "train_batch_size": sum(param_config["batch-size"].values())
            # * gradient_accumulation_steps,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": per_device_train_batch_size,
            "wall_clock_breakdown": False,
        }
        if bool(param_config["fp16-type"]):
            deepspeed["fp16"] = {
                "enabled": "auto",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        if param_config["bf16"]:
            deepspeed["bf16"] = {"enabled": True}
    else:
        deepspeed = None
    # initialize
    training_args = TrainingArguments(
        output_dir=model_dir,
        do_train=True,
        do_eval=False,  # default
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=param_config["learning-rate"],
        adam_beta1=0.9,  # same as BERT paper
        adam_beta2=0.98 if model_name == "roberta" else 0.999,  # same as BERT paper
        adam_epsilon=1e-6,
        weight_decay=0.01,  # same as BERT paper
        warmup_steps=param_config["warmup-steps"],
        logging_dir=os.path.join(os.path.dirname(__file__), f"runs/{run_name}"),
        save_steps=param_config["save-steps"]
        if "save-steps" in param_config.keys()
        else 50000,  # default:500
        save_strategy="steps",  # default:"steps"
        logging_steps=logging_steps,
        save_total_limit=20,  # optional
        seed=42,  # default
        bf16=param_config["bf16"],
        fp16=bool(param_config["fp16-type"] != 0),
        fp16_opt_level="O{}".format(param_config["fp16-type"]),
        # "O1":Mixed Precision (recommended for typical use),
        # "O2":“Almost FP16” Mixed Precision, "O3":FP16 training
        disable_tqdm=True,
        max_steps=param_config["train-steps"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=3,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        local_rank=local_rank,
        report_to="tensorboard",
        deepspeed=deepspeed,
    )
    if not do_continue:
        if local_rank != -1:
            training_args.per_device_train_batch_size = per_device_train_batch_size
        torch.save(training_args, os.path.join(model_dir, "training_args.bin"))

    # dataset
    dataset = datasets.load_from_disk(dataset_dir)
    dataset.set_format(type="torch")
    logger.info("Dataset is loaded")

    # model
    model: PreTrainedModel = get_model(
        model_name=model_name,
        tokenizer=tokenizer,
        load_pretrained=load_pretrained,
        param_config=param_config,
    )
    logger.info(f"{model_name} model is loaded")

    # data collator
    if model_name in ["bert", "roberta", "deberta", "debertav2"]:
        mlm_probability = 0.15
    elif model_name == "electra":
        mlm_probability = param_config["mask-percent"] / 100
    data_collator: Union[DataCollatorMixin, DataCollatorWithPadding]
    if is_dataset_masked:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=param_config["sequence-length"],
        )
    else:
        data_collator = utils.get_mask_datacollator(
            model_name=model_name,
            do_whole_word_mask=do_whole_word_mask,
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
        )
    logger.info("Datacollator was complete.")
    trainer = utils.MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        node_rank=node_rank,
    )
    trainer.batch_config = param_config["batch-size"]
    trainer.real_batch_size = sum(param_config["batch-size"].values())

    logger.info("Pretraining starts.")
    resume_from_checkpoint = True if do_continue else None
    trainoutput = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
        do_log_loss_gen_disc=bool(model_name == "electra"),
    )


def assert_config(param_config: dict, model_type: str, node_rank: int) -> bool:
    """
    Return
    model_name: str, bert, debertav2, electra, or roberta
    load_pretrained: bool, True when further pretrain and False when pretrain from scratch
    """
    if "electra-" in model_type.lower():
        model_name = "electra"
    elif "roberta-" in model_type.lower():
        model_name = "roberta"
    elif "debertav2-" in model_type.lower():
        model_name = "debertav2"
    elif "deberta-" in model_type.lower():
        model_name = "deberta"
    elif "bert-" in model_type.lower():
        model_name = "bert"
    else:
        raise ValueError("Argument model_type must contain electra or bert")
    if (
        len(
            set(param_config.keys())
            & {
                f"pretrained_{x}model_name_or_path"
                for x in ["", "generator_", "discriminator"]
            }
        )
        > 0
    ):
        load_pretrained = True
        set_assert = {"flozen-layers"}
        if model_name in ["bert", "roberta", "deberta", "debertav2"]:
            set_assert = set_assert | {"pretrained_model_name_or_path"}
            if model_name in ["deberta", "debertav2"]:
                set_assert = set_assert | {"pos_att_type"}
        if model_name == "electra":
            set_assert = set_assert | {
                f"pretrained_{x}_model_name_or_path"
                for x in ["generator", "discriminator"]
            }
    else:
        load_pretrained = False
        set_assert = {
            "number-of-layers",
            "hidden-size",
            "sequence-length",
            "ffn-inner-hidden-size",
            "attention-heads",
            "warmup-steps",
            "learning-rate",
            "batch-size",
            "train-steps",
            "fp16-type",
            "bf16"
        }
        if model_name == "electra":
            set_assert = set_assert | {
                "embedding-size",
                "generator-size",
                "mask-percent",
            }
    if param_config.keys() < set_assert:
        raise ValueError(
            f"{set_assert-param_config.keys()} is(are) not in parameter_file"
        )
    if str(node_rank) not in param_config["batch-size"].keys():
        raise ValueError(
            f"node_rank {node_rank} is not defined in batch-size of parameter_file"
        )
    logger.info(f"Config[{model_type}] is loaded")
    return model_name, load_pretrained


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # required
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="directory of corpus dataset"
    )
    parser.add_argument("--model_dir", type=str, required=True, help="directory to save pretrained model")
    parser.add_argument(
        "--parameter_file",
        type=str,
        required=True,
        help="json file defining model parameters",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="model parameter defined in the parameter_file. It must contain 'bert-', 'debertav2-', 'electra-', or 'roberta-'",
    )
    # optional
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--do_whole_word_mask", action="store_true")
    parser.add_argument("--do_continue", action="store_true")
    parser.add_argument(
        "--is_dataset_masked",
        action="store_true",
        help="use this option when masking process is already completed",
    )
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--deepspeed_bucket_size", type=float, default=5e8)
    parser.add_argument("--node_rank", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    utils.add_arguments_for_tokenizer(parser)
    args = parser.parse_args()
    assert not (
        args.subword_tokenizer_type == "sentencepiece" and args.do_whole_word_mask
    ), "Whole Word Masking cannot be applied with sentencepiece tokenizer"
    utils.assert_arguments_for_tokenizer(args)

    # global variables
    ram_gb: float = psutil.virtual_memory().total / 1073741824
    # dev_count: int = max(torch.cuda.device_count(), 1)
    # datasets.config.IN_MEMORY_MAX_SIZE = int(ram_gb * 0.9 / dev_count) * 10**9
    datasets.config.IN_MEMORY_MAX_SIZE = int(ram_gb * 0.9) * 10**9

    # parameter configuration
    with open(args.parameter_file, "r") as f:
        param_config = json.load(f)
    if args.model_type not in param_config:
        raise KeyError(f"{args.model_type} not in parameters.json")
    model_name, load_pretrained = assert_config(
        param_config[args.model_type], args.model_type, args.node_rank
    )

    tokenizer = utils.load_tokenizer(args)

    run_pretraining(
        tokenizer=tokenizer,
        dataset_dir=args.dataset_dir,
        model_name=model_name,
        model_dir=args.model_dir,
        load_pretrained=load_pretrained,
        param_config=param_config[args.model_type],
        do_whole_word_mask=args.do_whole_word_mask,
        do_continue=args.do_continue,
        is_dataset_masked=args.is_dataset_masked,
        use_deepspeed=args.use_deepspeed,
        deepspeed_bucket_size=args.deepspeed_bucket_size,
        node_rank=args.node_rank,
        local_rank=args.local_rank,
        run_name=args.run_name,
    )
