import torch
import torch.nn as nn

from seq2seq.models import EncoderRNN
from torch.autograd import Variable

class HiddenStateAnalysisEncoderRNN(EncoderRNN):

    KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS = 'hidden_activations_encoder'
    KEY_CELL_ACTIVATIONS_ALL_TIMESTEPS = 'cell_activations_encoder'

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        # TODO allow larger batches
        if not input_var.size()[0] == 1: raise ValueError('Batch size must be 1')
        batch_size = input_var.size()[0]

        # TODO allow bidirectional RNNs
        if self.rnn.bidirectional: raise ValueError('RNN must not be bidirectional')
        self.num_directions = 1

        #init hidden state with zeros
        hidden_states = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size))

        if self.rnn_cell == nn.LSTM:
            cell_states = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size))

        hidden_activations_all_timesteps = []
        cell_activations_all_timesteps = []
        full_output = torch.zeros(batch_size, input_var.size()[1], self.hidden_size)

        for i in range(0,embedded.shape[1]):
            input = embedded[:,i,:].view(1,1,-1) # shape of input: seq_len, batch, input_size)

            if self.rnn_cell == nn.LSTM:
                output, (hidden_states, cell_states) = self.rnn(input, (hidden_states, cell_states))
                cell_activations_all_timesteps.append(cell_states)
            else:
                output, hidden_states = self.rnn(input, hidden_states)

            full_output[:, i, :] = output
            hidden_activations_all_timesteps.append(hidden_states)


        ret_dict = dict()
        ret_dict[HiddenStateAnalysisEncoderRNN.KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS] = hidden_activations_all_timesteps

        if self.rnn_cell == nn.LSTM:
            ret_dict[HiddenStateAnalysisEncoderRNN.KEY_CELL_ACTIVATIONS_ALL_TIMESTEPS] = cell_activations_all_timesteps

        return full_output, hidden_states, ret_dict

