import logging
import os
import argparse

import torch
import torchtext

import seq2seq
from seq2seq.loss import Perplexity, AttentionLoss, NLLLoss
from seq2seq.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField, AttentionField
from seq2seq.trainer import SupervisedTrainer
from seq2seq.evaluator import Evaluator


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	parser = argparse.ArgumentParser()

	parser.add_argument('--checkpoint-path', help='Give the checkpoint path from which to load the model')
	parser.add_argument('--test_data', help='Path to test data')
	parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
	parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
	parser.add_argument('--log-level', default='info', help='Logging level.')

	#TODO are these two arguments even necessary?
	parser.add_argument('--attention', choices=['pre-rnn', 'post-rnn'], default=False)
	parser.add_argument('--attention_method', choices=['dot', 'mlp', 'hard'], default=None)

	parser.add_argument('--use-attention-loss', action='store_true')
	parser.add_argument('--scale_attention_loss', type=float, default=1.)

	parser.add_argument('--ignore-output-eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

	opt = parser.parse_args()


	LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
	logging.info(opt)

	IGNORE_INDEX = -1
	output_eos_used = not opt.ignore_output_eos

	if not opt.attention and opt.attention_method:
	    parser.error("Attention method provided, but attention is not turned on")

	if opt.attention and not opt.attention_method:
	    parser.error("Attention turned on, but no attention method provided")

	if opt.use_attention_loss and opt.attention_method == 'hard':
	    parser.warning("Did you mean to use attention loss in combination with hard attention method?")

	if torch.cuda.is_available():
	    logging.info("Cuda device set to %i" % opt.cuda_device)
	    torch.cuda.set_device(opt.cuda_device)

	# load model
	logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
	checkpoint = Checkpoint.load(opt.checkpoint_path)
	seq2seq = checkpoint.model
	input_vocab = checkpoint.input_vocab
	output_vocab = checkpoint.output_vocab


	# Prepare dataset and loss
	src = SourceField()
	tgt = TargetField(output_eos_used)

	tabular_data_fields = [('src', src), ('tgt', tgt)]

	if opt.use_attention_loss or opt.attention_method == 'hard':
	  attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
	  tabular_data_fields.append(('attn', attn))

	src.vocab = input_vocab
	tgt.vocab = output_vocab
	tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
	tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]
	max_len = opt.max_len

	def len_filter(example):
	    return len(example.src) <= max_len and len(example.tgt) <= max_len

	# generate test set
	test = torchtext.data.TabularDataset(
	    path=opt.test_data, format='tsv',
	    fields=tabular_data_fields,
	    filter_pred=len_filter
	)

	# When chosen to use attentive guidance, check whether the data is correct for the first
	# example in the data set. We can assume that the other examples are then also correct.
	if opt.use_attention_loss or opt.attention_method == 'hard':
	    if len(test) > 0:
	        if 'attn' not in vars(test[0]):
	            raise Exception("AttentionField not found in test data")
	        tgt_len = len(vars(test[0])['tgt']) - 1 # -1 for SOS
	        attn_len = len(vars(test[0])['attn']) - 1 # -1 for preprended ignore_index
	        if attn_len != tgt_len:
	            raise Exception("Length of output sequence does not equal length of attention sequence in test data.")



	data_func = SupervisedTrainer.get_batch_data

	run_model_on_test_data(model=seq2seq, data=test, get_batch_data=data_func)


def run_model_on_test_data(model, data, get_batch_data):
        # create batch iterator
        iterator_device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=1,
            sort=True, sort_key=lambda x: len(x.src),
            device=iterator_device, train=False)

        # loop over test data
        with torch.no_grad():
            for batch in batch_iterator:
                input_variable, input_lengths, target_variable = get_batch_data(batch)

                ##TODO define own forward path to get hidden states for all timesteps

                decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths.tolist(), target_variable)

                
                print('\n\n\n',decoder_hidden)
                print(decoder_hidden.shape)
                return

        return


if __name__ == '__main__':
    main()