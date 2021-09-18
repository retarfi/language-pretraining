# language-pretraining

## Train Tokenizer
```
python train_tokenizer.py \
--input_file ./share/corpus.txt \
--model_dir ./share/tokenizer \
--intermediate_dir ./data/corpus_split/ \
--num_files 20 \
--mecab_dic_type ipadic \
--tokenizer_type wordpiece \
--vocab_size 32768 \
--min_frequency 2 \
--limit_alphabet 6129 \
--num_unused_tokens 10 
```
- `intermediate_dir`は中間生成ファイルのディレクトリを指定
- `num_files`は並列処理の分割数(これにより中間生成ファイルの数も決まる)を指定 推奨はマシンのスレッド数
- `limit_alphabet`は1文字の語彙数を指定(defaultは東北大BERTと同じ)
- `num_unused_tokens`はファインチューニング時に追加できる空きの語彙数を指定(defaultは東北大BERTと同じ)

## Pretraining
1 nodeで行う場合:
```
python run_pretraining.py \
--input_file ./share/corpus.txt \
--tokenizer_dir ./share/tokenizer/ \
--model_dir ./model/bert/ \
--parameter_file parameter.json \
--model_type bert-base \
--fp16_type 0 \
--tokenizer_type wordpiece \
--mecab_dic_type ipadic \
--do_whole_word_mask \
(--do_continue \)
(--disable_overwrite_cache)
```

multi nodeで行う場合:
```
NCCL_SOCKET_IFNAME=hoge CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
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
--do_whole_word_mask \
(--do_continue \)
(--disable_overwrite_cache)
```

- `model_type`はparameter.json内にあるモデルの種類を指定
- `fp16_type`はAoex AMPのoptimization levelを指定する(詳細はhttps://nvidia.github.io/apex/amp.html参照)
- `do_continue`は途中から学習を継続したい場合に指定
- `disable_overwrite_cache`は学習を継続したい場合に再度学習データのcacheを作成しなおさない場合に指定
- `NCCL_SOCKET_IFNAME`はNICのカードを指定する
- `CUDA_VISIBLE_DEVICES`は使うGPUを指定する(指定しない場合は全てのGPUに分配される)
- `node_rank`は各ノードの順位を指定する(3台なら0,1,2)
- `local_rank`は各ノードにおけるプロセス順位(1ノードにつき1プロセスなら0)


### ELECTRA
run_pretraining.pyによって生成されるモデルは、GeneratorとDiscriminatorの両方が含まれるため、分離することが必要。
```
python extract_electra_model.py \
--input_dir ./model/electra/checkpoint-1000000 \
--output_dir ./model/electra/extracted-1000000 \
--parameter_file parameter.json \
--model_type electra-small \
--generator \
--discriminator
```

この場合、./model/electra/extracted-1000000/generator/にgeneratorモデルが、./model/electra/extracted-1000000/discriminator/にdiscriminatorモデルが保存される


## parameter.json
### batch_sizeについて
single nodeの場合: keyを-1、valueにバッチサイズを指定
```
"batch-size" : {"-1" : 128},
```

multi nodeの場合、run_pretraining.pyの`--node_rank`に対応するノード番号ごとにバッチサイズを指定
```
"batch-size" : {
    "0" : 100,
    "1" : 100,
    "2" : 56
},
```



## Tensorboard
```
tensorboard --logdir ./runs/
```
で実行される
