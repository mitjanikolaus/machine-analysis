import torch
import torch.nn as nn

from seq2seq.models import EncoderRNN
from torch.autograd import Variable

class HiddenStateAnalysisEncoderRNN(EncoderRNN):

    KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS = 'hidden_activations_encoder'

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        #TODO assuming batch size of 1
        batch_size = 1
        input_for_rnn = embedded[0]

        #TODO better init?
        self.num_directions = 1
        hidden = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size))

        hidden_activations_all_timesteps = []

        for i in range(0,input_for_rnn.shape[0]):
            # shape of input: seq_len, batch, input_size)
            input = input_for_rnn[i].view(1, 1, -1)

            output, hidden = self.rnn(input, hidden)

            hidden_activations_all_timesteps.append(hidden)

        ret_dict = dict()
        ret_dict[HiddenStateAnalysisEncoderRNN.KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS] = hidden_activations_all_timesteps

        #TODO uncommented because of error
        #if self.variable_lengths:
        #    output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden, ret_dict

