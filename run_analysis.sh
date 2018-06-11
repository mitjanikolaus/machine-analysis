#! /bin/sh

CHECKPOINT_PATH='../machine-zoo/guided/gru/1/'
TEST_DATA='../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/heldout_tables.tsv'

ATTENTION='pre-rnn'
ATTENTION_METHOD='mlp'

python get_hidden_activations.py --use-attention-loss --ignore-output-eos --test_data $TEST_DATA --checkpoint-path $CHECKPOINT_PATH --attention_method $ATTENTION_METHOD --attention $ATTENTION