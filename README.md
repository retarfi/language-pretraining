# language-pretraining

## Train Tokenizer
```
python train_tokenizer.py \
--input_file_or_dir=./data/materials/corpus_tokenized/ \
--output_dir=./model/tokenizer \
--tokenizer_type=wordpiece \
--vocab_size=32000 \
--min_frequency=2 \
```

### Pretraining
1 nodeで行う場合:
```
python run_pretraining.py \
--input_file=./data/materials/corpus.txt \
--tokenizer_dir=./model/tokenizer/ \
--model_dir=./model/bert/ \
--parameter_file parameter.json \
--model_type base \
--tokenizer_type=wordpiece \
--mecab_dic_type=ipadic \
--do_whole_word_mask \
(--do_continue)
```

multi nodeで行う場合:
```
NCCL_SOCKET_IFNAME=hoge CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.0.0.1" \
--master_port=50916 run_pretraining.py \
--input_file ./data/materials/small.txt \
--tokenizer_dir ./model/tokenizer/ \
--model_dir ./model/bert/ \
--parameter_file parameter.json \
--model_type mybase-test \
--tokenizer_type wordpiece \
--mecab_dic_type ipadic \
--do_whole_word_mask \
--local_rank 0 \
(--do_continue)
```

## Tensorboard
```
tensorboard --logdir ./runs/
```
で実行される
