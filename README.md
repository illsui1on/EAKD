# EAKD

This replication package contains the adversarial attack pre-trained model, GA-guided model reduction, and source code for training.

## Environment configuration


GraphCodeBERT need a parser to extract data flows from the source code, please go to `./parser` to compile the parser first. Pls run:
```
cd parser
bash build.sh
```

## KD Results
To facilitate researchers to reproduce our experiments, we provide solution results for the model search space.:

```
3MB:{'attention_heads': 8, 'hidden_dim': 96, 'intermediate_size': 64, 'n_layers': 12, 'vocab_size': 1000}

10MB:{'attention_heads': 8, 'hidden_dim': 112, 'intermediate_size': 128, 'n_layers': 12, 'vocab_size': 13000}

25MB:{'attention_heads': 8, 'hidden_dim': 112, 'intermediate_size': 128, 'n_layers': 11, 'vocab_size': 36000}
 ```
 
## Dataset Collection
RKD requires the collection of adversarial examples, and in this study, we chose to use ALERT as the method for collecting 
adversarial examples. For details on using ALERT, please refer to "Natural Attack for Pre-trained Models of Code."


## EAKD Tarining

Once the data is prepared, we can proceed with the final EAKD training. For example, in the path /CodeT5/vulnerability_prediction/distill, we execute the following command to complete the model training:

```
python3 distill.py \
    --do_train \
    --train_data_file=../dataset/vulnerability_prediction/adv_train.jsonl \
    --eval_data_file=../dataset/vulnerability_prediction/adv_valid.jsonl \
    --model_dir ../checkpoint \
    --size 3 \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 20 \
 ```
 
