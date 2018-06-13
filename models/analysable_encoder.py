
# STD
from collections import defaultdict

# EXT
import torch
import torch.nn as nn
from torch.autograd import Variable

# PROJECT
from seq2seq.models import EncoderRNN
from models.analysable_cells import AnalysableLSTMCell, AnalysableGRUCell, AnalysableCellsMixin


class HiddenStateAnalysisEncoderRNN(EncoderRNN, AnalysableCellsMixin):
    """
    Modified version of the RNN Encoder which allows to store the networks activations for further analysis.
    """
    KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS = 'hidden_activations_encoder'
    KEY_CELL_ACTIVATIONS_ALL_TIMESTEPS = 'cell_activations_encoder'

    def __init__(self, *args, **kwargs):
        EncoderRNN.__init__(self, *args, **kwargs)
        AnalysableCellsMixin.__init__(self, *args, **kwargs)

    def forward(self, input_var, input_lengths=None):
        return_dict = dict()
        hidden_activations_all_timesteps = []
        cell_activations_all_timesteps = []
        gate_activations_all_timesteps = defaultdict(list)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        # Replace cells once
        self.replace_cells()

        # TODO allow larger batches
        if not input_var.size()[0] == 1: raise ValueError('Batch size must be 1')
        batch_size = input_var.size()[0]

        # TODO allow bidirectional RNNs
        if self.rnn.bidirectional: raise ValueError('RNN must not be bidirectional')
        self.num_directions = 1

        #init hidden state with zeros
        hidden_states = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size))

        # Replace RNN cells with their analyzable counterparts which store activations
        if self.rnn_cell == AnalysableLSTMCell:
            cell_states = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size))
            hidden = (hidden_states, cell_states)

        elif self.rnn_cell == AnalysableGRUCell:
            hidden = hidden_states

        full_output = torch.zeros(batch_size, input_var.size()[1], self.hidden_size)

        for i in range(0, embedded.shape[1]):
            input = embedded[:, i, :].view(1, 1, -1)  # shape of input: seq_len, batch, input_size)

            output, hidden, gates = self.rnn(input, hidden)

            # LSTM case
            if self.rnn_cell == AnalysableLSTMCell:
                cell_activations_all_timesteps.append(hidden[1])

            full_output[:, i, :] = output
            hidden_activations_all_timesteps.append(hidden_states)

            for gate_name, activations in gates.items():
                gate_activations_all_timesteps[gate_name].append(activations)

        return_dict[HiddenStateAnalysisEncoderRNN.KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS] = hidden_activations_all_timesteps

        if self.rnn_cell == AnalysableLSTMCell:
            return_dict[HiddenStateAnalysisEncoderRNN.KEY_CELL_ACTIVATIONS_ALL_TIMESTEPS] = cell_activations_all_timesteps

        for gate_name, all_activations in gate_activations_all_timesteps.items():
            return_dict[gate_name + "_encoder"] = all_activations

        return full_output, hidden_states, return_dict
