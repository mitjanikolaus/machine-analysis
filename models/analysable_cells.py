"""
Define versions of RNN cells which enable the user to record the activations.
"""

# EXT
import torch.nn.functional as F
from torch import nn


class AnalysableCellsMixin:
    def __init__(self, *args, **kwargs):
        self.cells_replaced = False

    def replace_cells(self):
        """
        Replace the RNN cells with their analysable counterparts once during the first forward pass.
        """
        if not self.cells_replaced:
            # Replace RNN cells with their analyzable counterparts which store activations
            if self.rnn_cell == nn.LSTM:
                self.rnn_cell = AnalysableLSTMCell

                # Init new cell with trained weights
                self.rnn = AnalysableLSTMCell(
                    num_layers=self.rnn.num_layers, batch_first=self.rnn.batch_first,
                    w_hh=self.rnn.weight_hh_l0, w_ih=self.rnn.weight_ih_l0, b_hh=self.rnn.bias_hh_l0,
                    b_ih=self.rnn.bias_ih_l0
                )

            elif self.rnn_cell == nn.GRU:
                self.rnn_cell = AnalysableGRUCell

                # Init new cell with trained weights
                self.rnn = AnalysableGRUCell(
                    num_layers=self.rnn.num_layers, batch_first=self.rnn.batch_first,
                    w_hh=self.rnn.weight_hh_l0, w_ih=self.rnn.weight_ih_l0, b_hh=self.rnn.bias_hh_l0,
                    b_ih=self.rnn.bias_ih_l0
                )

            self.cells_replaced = True


class AnalysableLSTMCell(nn.Module):
    def __init__(self, num_layers, w_ih, w_hh, b_ih=None, b_hh=None, batch_first=True):
        super().__init__()
        self.bidirectional = False
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Use weights from an already trained RNN
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = F.linear(input, self.w_ih, self.b_ih) + F.linear(hx, self.w_hh, self.b_hh)

        ingate, forgetgate, cy_tilde, outgate = gates.chunk(4, 2)  #dim modified from 1 to 2

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cy_tilde = F.tanh(cy_tilde)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cy_tilde)
        hy = outgate * F.tanh(cy)

        return (hy, cy), {
            "input_gate_activations": ingate, "forget_gate_activations": forgetgate,
            "output_gate_activations": outgate, "cell_gate_activations": cy_tilde
        }


class AnalysableGRUCell(nn.Module):
    def __init__(self, num_layers, w_ih, w_hh, b_ih=None, b_hh=None, batch_first=True):
        super().__init__()
        self.bidirectional = False
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Use weights from an already trained RNN
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def forward(self, input, hidden):
        gi = F.linear(input, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy, {
            "input_gate_activations": inputgate, "reset_gate_activations": resetgate, "new_gate_activations": newgate
        }
