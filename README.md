# BERT and ELECTRA Models for Japanese

This is a repository of pretrained Japapanese BERT and ELECTRA models.
The models are available in Transformers by Hugging Face: [https://huggingface.co/izumi-lab](https://huggingface.co/izumi-lab).
BERT-small, ELECTRA-small, and ELECTRA-small-paper models trained by Wikipedia or financial dataset is available in this URL.
BERT-base model trained by financial dataset will be available in the future.

## Model Architecture
The architecture of BERT-small and ELECTRA-small-paper models are the same as those in [the original ELECTRA paper](https://arxiv.org/abs/2003.10555) (ELECTRA-small-paper is described as ELECTRA-small in the paper).
The architecture of ELECTRA-small is the same as that in [the ELECTRA implementation by Google](https://github.com/google-research/electra).

| Parameter | BERT-small | ELECTRA-small | ELECTRA-small-paper |
| :---: | :---: | :---: | :---: |
| Number of layers | 12 | 12 | 12 |
| Hidden Size | 256 | 256 | 256 |
| Attention Heads | 4 | 4 | 4 |
| Generator Size | - | 1/1 | 1/4 |
| Train Steps | 1.45M | 1M | 1M |

Other models such as BERT-base or ELECTRA-base are also available in this implementation.
You can also add your original parameters in parameter.json.


## Training Data
Training data are aggregated to a text file.
Each sentence is in one line and a blank line is inserted between documents.

### Wikipedia Model
The normal models (not financial models) are trained on the Japanese version of Wikipedia, using [Wikipedia dump](https://dumps.wikimedia.org/jawiki/) file as of June 1, 2021.
The corpus file is 2.9GB, consisting of approximately 20M sentences.

### Financial Model
The financial models are trained on Wikipedia corpus and financial corpus.
The Wikipedia corpus is the same as described above.
The financial corpus consists of 2 corpora:
- Summaries of financial results from October 9, 2012, to December 31, 2020
- Securities reports from February 8, 2018, to December 31, 2020
The financial corpus file is 5.2GB, consisting of approximately 27M sentences.

## Usage
### Train Tokenizer
In our pretrained models, the texts are first tokenized by [MeCab](https://taku910.github.io/mecab/) with [IPAdic](https://pypi.org/project/ipadic/) dictionary and then split into subwords by the WordPiece algorithm.
For MeCab dictionary, [unidic](https://github.com/polm/unidic-lite) and unidic-lite are also available.
[Sentencepiece](https://github.com/google/sentencepiece) is also available for subword algorithm, but we do not validate performance.


```
$ python train_tokenizer.py \
--input_file corpus.txt \
--model_dir tokenizer/ \
--intermediate_dir ./data/corpus_split/ \
--num_files 20 \
--mecab_dic_type ipadic \
--tokenizer_type wordpiece \
--vocab_size 32768 \
--min_frequency 2 \
--limit_alphabet 6129 \
--num_unused_tokens 10 
```


### Training

Distributed training is available.
For run command, please see the [PyTorch document](https://pytorch.org/docs/stable/distributed.html#launch-utility) in detail.
In official PyTorch implementation, different batch size between nodes is not available.
We improved PyTorch sampling implementation (utils/trainer_pt_utils.py).

For example, `bert-base-dist` model is defined in parameter.json:
```
"bert-base-dist" : {
    "number-of-layers" : 12,
    "hidden-size" : 768,
    "sequence-length" : 512,
    "ffn-inner-hidden-size" : 3072,
    "attention-heads" : 12,
    "warmup-steps" : 10000,
    "learning-rate" : 1e-4,
    "batch-size" : {
        "0" : 80,
        "1" : 80,
        "2" : 48,
        "3" : 48
    },
    "train-steps" : 1000000,
    "save-steps" : 50000,
    "logging-steps" : 5000
}
```
In this case, node 0 and node 1 have 80 batch sizes and node 2 and node 3 have 48 respectively.
If node 0 has 2 GPUs, each GPU have a 40 batch size.
**10G or higher network speed** is recommended for training with multi-nodes.

`fp16_type` argument specifies which precision mode to use:

- 0: FP32 training
- 1: Mixed Precision
- 2: "Almost FP16" Mixed Precision
- 3: FP16 training

In detail, please see [NVIDIA Apex document](https://nvidia.github.io/apex/amp.html).

The whole word masking option is also available.

```
# Train with 1 node
$ python run_pretraining.py \
--input_file ./share/corpus.txt \
--tokenizer_dir ./share/tokenizer/ \
--model_dir ./model/bert/ \
--parameter_file parameter.json \
--model_type bert-base \
--fp16_type 0 \
--tokenizer_type wordpiece \
--mecab_dic_type ipadic \
(--do_whole_word_mask \)
(--do_continue \)
(--disable_overwrite_cache)

# Train with multi-node and multi-process
$ NCCL_SOCKET_IFNAME=eno1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.0.0.1" \
--master_port=50916 run_pretraining.py \
--input_file ./share/corpus.txt \
--tokenizer_dir ./share/tokenizer/ \
--model_dir ./model/bert/ \
--parameter_file parameter.json \
--model_type bert-base \
--fp16_type 0 \
--tokenizer_type wordpiece \
--mecab_dic_type ipadic \
--node_rank 0 \
--local_rank 0 \
(--do_whole_word_mask \)
(--do_continue \)
(--disable_overwrite_cache)
```


### ELECTRA
ELECTRA models generated by run_pretraining.py contain both generator and discriminator.
For general use, separation is needed.

```
$ python extract_electra_model.py \
--input_dir ./model/electra/checkpoint-1000000 \
--output_dir ./model/electra/extracted-1000000 \
--parameter_file parameter.json \
--model_type electra-small \
--generator \
--discriminator
```

In this example, the generator model is saved in `./model/electra/extracted-1000000/generator/` and discriminator model is saved in `./model/electra/extracted-1000000/discriminator/` respectively.

### Training Log
Tensorboard is available for the training log.

## Citation
### Pretrained Model
**There will be another paper for this pretrained model.
Be sure to check here again when you cite.**
```
@inproceedings{bert_electra_japanese,
  title = {Construction and Validation of a Pre-Trained Language Model
Using Financial Documents}
  author = {Masahiro Suzuki and Hiroki Sakaji and Masanori Hirano and Kiyoshi Izumi},
  month = {oct},
  year = {2021},
  booktitle = {"Proceedings of JSAI Special Interest Group on Financial Infomatics (SIG-FIN) 27"}
}
```

### This Implementation
```
@misc{bert_electra_japanese,
  author = {Masahiro Suzuki},
  title = {BERT and ELECTRA Models for Japanese},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/retarfi/language-pretraining}}
}
```


## Licenses
The pretrained models are distributed under the terms of the [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The codes in this repository are distributed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).


## Related Work
- Original BERT model by Google Research Team
    - https://github.com/google-research/bert
- Original ELECTRA model by Google Research Team
    - https://github.com/google-research/electra
- Pretrained Japanese BERT models
    - Autor Tohoku University
    - https://github.com/cl-tohoku/bert-japanese
- ELECTRA training with PyTorch implementation
    - Author: Richard Wang
    - https://github.com/richarddwang/electra_pytorch

## Acknowledgement
This work was supported by JSPS KAKENHI Grant Number JP21K12010. 
