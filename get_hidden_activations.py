"""
Define a function to retrieve a wide variety of activations for later analysis.
"""

# STD
import logging
import os
from collections import defaultdict

# EXT
import numpy as np
import torch
import torchtext
from seq2seq.dataset import SourceField, TargetField, AttentionField
from seq2seq.trainer import SupervisedTrainer

# from models.AnalysableSeq2seq import AnalysableSeq2seq
# from models.HiddenStateAnalysisDecoderRNN import HiddenStateAnalysisDecoderRNN
# from models.HiddenStateAnalysisEncoderRNN import HiddenStateAnalysisEncoderRNN
from activations import ActivationsDataset, CounterDataset
from models.analysable_decoder import HiddenStateAnalysisDecoderRNN
from models.analysable_encoder import HiddenStateAnalysisEncoderRNN
from models.analysable_seq2seq import AnalysableSeq2seq
from models.analysable_cells import AnalysableLSTMCell

def run_and_get_hidden_activations(checkpoint_path, test_data_path, attention_method, use_attention_loss,
                                   ignore_output_eos, max_len=50, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))


    IGNORE_INDEX = -1
    output_eos_used = not ignore_output_eos

    # load model
    logging.info("loading checkpoint from {}".format(os.path.join(checkpoint_path)))
    checkpoint = AnalysableSeq2seq.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    # Prepare dataset and loss
    src = SourceField()
    tgt = TargetField(output_eos_used)

    tabular_data_fields = [('src', src), ('tgt', tgt)]

    if use_attention_loss or attention_method == 'hard':
      attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
      tabular_data_fields.append(('attn', attn))

    src.vocab = input_vocab
    tgt.vocab = output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    # generate test set
    test = torchtext.data.TabularDataset(
        path=test_data_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )

    # When chosen to use attentive guidance, check whether the data is correct for the first
    # example in the data set. We can assume that the other examples are then also correct.
    if use_attention_loss or attention_method == 'hard':
        if len(test) > 0:
            if 'attn' not in vars(test[0]):
                raise Exception("AttentionField not found in test data")
            tgt_len = len(vars(test[0])['tgt']) - 1 # -1 for SOS
            attn_len = len(vars(test[0])['attn']) - 1 # -1 for preprended ignore_index
            if attn_len != tgt_len:
                raise Exception("Length of output sequence does not equal length of attention sequence in test data.")

    data_func = SupervisedTrainer.get_batch_data

    activations_dataset = run_model_on_test_data(model=seq2seq, data=test, get_batch_data=data_func)

    if save_path is not None:
        activations_dataset.save(save_path)


def run_model_on_test_data(model, data, get_batch_data):
        # Store activations and inputs for later
        all_input_seqs = []
        all_encoder_activations = []
        all_decoder_activations = []
        all_model_outputs = []
        all_model_activations = defaultdict(list)

        # create batch iterator
        iterator_device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=1,
            sort=True, sort_key=lambda x: len(x.src),
            device=iterator_device, train=False)

        # loop over test data
        with torch.no_grad():
            for i, batch in enumerate(batch_iterator):

                # TODO: use for getting smaller datasets for debugging
                # if i > 10: break

                input_variable, input_lengths, target_variable = get_batch_data(batch)

                # using own forward path to get hidden states for all timesteps
                decoder_outputs, decoder_hidden, ret_dict_decoder, ret_dict_encoder = model(input_variable, input_lengths.tolist(), target_variable)


                hidden_activations_encoder = ret_dict_encoder[HiddenStateAnalysisEncoderRNN.KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS]
                hidden_activations_decoder = ret_dict_decoder[HiddenStateAnalysisDecoderRNN.KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS]

                all_input_seqs.append(input_variable)
                all_encoder_activations.append(hidden_activations_encoder)
                all_decoder_activations.append(hidden_activations_decoder)
                all_model_outputs.append(all_decoder_activations)

                for timestep in range(len(hidden_activations_encoder)):
                   hidden_activations_encoder[timestep] = hidden_activations_encoder[timestep].numpy()

                # hidden_activations_encoder = np.array(hidden_activations_encoder).reshape(4, 512)

                #plot first 100 hidden activations
                # hidden_activations_encoder = hidden_activations_encoder[:,0:100]
                decoder_outputs, decoder_hidden, return_dict_decoder, return_dict_encoder = model(
                    input_variable, input_lengths.tolist(), target_variable
                )
                current_sample_model_activations = dict(return_dict_encoder)
                current_sample_model_activations.update(return_dict_decoder)

                # Store activations
                for activations_name, activations in current_sample_model_activations.items():
                    if "activations" in activations_name:
                        all_model_activations[activations_name].append(activations)

        # dataset = ActivationsDataset(all_input_seqs, all_model_outputs, **all_model_activations)


        gate_activations = {key: all_model_activations[key] for key in filter(lambda key: 'gate' in key, all_model_activations.keys())}
        gate_dataset = ActivationsDataset(all_input_seqs, all_model_outputs, **gate_activations)

        counter_dataset = CounterDataset(
            all_input_seqs,
            all_model_outputs,
            encoder_activations=all_encoder_activations,
            decoder_activations=all_decoder_activations
        )


        return gate_dataset

test_data = '../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/heldout_tables.tsv'

cell_types = ['gru', 'lstm']
model_types = ['guided', 'baseline']
# model_runs = range(1, 5+1)
model_runs = [1]
checkpoint_template = '../machine-zoo/{}/{}/{}/'
dataset_name_model_part = '{}_{}_run_{}'

checkpoints = [
    (checkpoint_template.format(model_type, cell_type, run), dataset_name_model_part.format(model_type, cell_type, run))
        for run in model_runs
        for cell_type in cell_types
        for model_type in model_types
]

dataset_template = '../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/{}/sample{}/{}{}.tsv'
dataset_name_data_part = 'sample_{}_{}_{}'
lengths = ['samples', 'longer_compositions']
# sample_numbers = range(1, 5+1)
sample_numbers = [1]
tasks = ['heldout_tables', 'heldout_compositions', 'new_compositions']

datasets = [
    (dataset_template.format(length, sample_number, task, '' if length == 'samples' else '4'), dataset_name_data_part.format(sample_number, task, 'simple' if length == 'samples' else length))
        for length in lengths
        for sample_number in sample_numbers
        for task in tasks
]

for checkpoint, name_model_part in checkpoints:
    for dataset, name_data_part in datasets:

        print('data/{}_{}.pt'.format(name_model_part, name_data_part))

        # if 'lstm' in name_model_part.lower():
        if True:
            run_and_get_hidden_activations(
                checkpoint,
                dataset,
                attention_method='mlp',
                use_attention_loss=True,
                ignore_output_eos=True,
                save_path='data/gate_datasets/{}_{}.pt'.format(name_model_part, name_data_part)
            )


# checkpoint_path = '../machine-zoo/guided/gru/1/'
# test_data = '../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/heldout_tables.tsv'
#
# run_and_get_hidden_activations(
#     checkpoint_path, test_data, attention_method='mlp', use_attention_loss=True, ignore_output_eos=True,
#     save_path="./test_activations_gru_1_heldout_tables.pt"
# )
