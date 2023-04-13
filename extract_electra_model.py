import argparse
import copy
import json
import logging
import os
import warnings
from fractions import Fraction
from typing import Dict, Optional, Set, Union

import transformers
from transformers import AutoConfig, DebertaV2Config, ElectraConfig

transformers.logging.set_verbosity_info()
transformers.logging.enable_explicit_format()

from models import ElectraForPretrainingModel, DebertaV3ForPreTraining
from utils.logger import make_logger_setting


warnings.simplefilter("ignore", UserWarning)
transformers.utils.check_min_version("4.10.2")


def main(
    input_dir: str,
    output_dir: str,
    param_config: Optional[Dict[str, Union[bool, int, str]]] = None,
    save_generator: bool = False,
    save_discriminator: bool = False,
) -> None:
    config_discriminator = AutoConfig.from_pretrained(input_dir)
    if config_discriminator.architectures[0] == "ElectraForPretrainingModel":
        assert isinstance(config_discriminator, ElectraConfig)
        if "generator-size" in param_config.keys():
            frac_generator = Fraction(param_config["generator-size"])
            config_generator = ElectraConfig(
                vocab_size=config_discriminator.vocab_size,
                embedding_size=param_config["embedding-size"],
                hidden_size=int(param_config["hidden-size"] * frac_generator),
                num_attention_heads=int(param_config["attention-heads"] * frac_generator),
                num_hidden_layers=param_config["number-of-layers"],
                intermediate_size=int(
                    param_config["ffn-inner-hidden-size"] * frac_generator
                ),
            )
        elif "pretrained_generator_model_name_or_path" in param_config.keys():
            config_generator = ElectraConfig.from_pretrained(
                param_config["pretrained_generator_model_name_or_path"]
            )
        else:
            raise ValueError(
                "For generator, pretrained_generator_model_name_or_path or <generator-size, embedding-size, hidden-size, attention-heads, number-of-layers, and ffn-inner-hidden-size> must be specified in json file"
            )

        model = ElectraForPretrainingModel.from_pretrained_together(
            input_dir,
            config_generator=config_generator,
            config_discriminator=config_discriminator,
        )
        generator = model.generator
        discriminator = model.discriminator
    elif config_discriminator.architectures[0] == "DebertaV3ForPreTraining":
        assert isinstance(config_discriminator, DebertaV2Config)
        config_generator = copy.deepcopy(config_discriminator)
        config_generator.num_hidden_layers = config_generator.num_hidden_layers // 2
        model = DebertaV3ForPreTraining.from_pretrained_together(
            input_dir,
            config_generator=config_generator,
            config_discriminator=config_discriminator
        )
        generator = model.generator.deberta
        discriminator = model.discriminator.deberta
    else:
        raise ValueError(f"Got invalid architectures {config_discriminator.architectures[0]}")
    

    if save_generator:
        save_path = os.path.join(output_dir, "generator")
        generator.save_pretrained(save_path)
        logger.info(f"Generator is saved at {save_path}")
    if save_discriminator:
        save_path = os.path.join(output_dir, "discriminator")
        discriminator.save_pretrained(save_path)
        logger.info(f"Discriminator is saved at {save_path}")


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--parameter_file", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument(
        "--generator", action="store_true", help="Extract generator model if enabled"
    )
    parser.add_argument(
        "--discriminator",
        action="store_true",
        help="Extract discriminator model if enabled",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    make_logger_setting(logger)

    param_config: Optional[Dict[str, Union[bool, int, str]]] = None
    if args.parameter_file is not None and args.model_type is not None:
        # parameter configuration
        with open(args.parameter_file, "r") as f:
            param_config = json.load(f)
        if args.model_type not in param_config:
            raise KeyError(f"{args.model_type} not in parameters.json")
        param_config = param_config[args.model_type]
        set_assert: Set[str] = {
            "number-of-layers",
            "hidden-size",
            "sequence-length",
            "ffn-inner-hidden-size",
            "attention-heads",
            "warmup-steps",
            "learning-rate",
            "batch-size",
            "train-steps",
            "embedding-size",
            "generator-size",
            "mask-percent",
        }
        if param_config.keys() < set_assert:
            raise ValueError(
                f"{set_assert-param_config.keys()} is(are) not in parameter_file"
            )
        logger.info(f"Config[{args.model_type}] is loaded")
    else:
        if args.parameter_file is None:
            logger.warning("args.parameter_file is ignored because args.model_type is not specified")
        if args.model_type is None:
            logger.warning("args.model_type is ignored because args.parameter_file is not specified")

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        param_config=param_config,
        save_generator=args.generator,
        save_discriminator=args.discriminator,
    )
