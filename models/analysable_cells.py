"""
Define versions of RNN cells which enable the user to record the activations.
"""

# EXT
import torch.nn.functional as F
from torch import nn


class AnalysableCellsMixin:
    """
    Add a function to replace RNN cells with their analysable counterparts to a class.
    """
    def __init__(self, *args, **kwargs):
        self.cells_replaced = False

    def replace_cells(self, save_dont_return=False):
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
                    b_ih=self.rnn.bias_ih_l0, save_dont_return=save_dont_return
                )

            elif self.rnn_cell == nn.GRU:
                self.rnn_cell = AnalysableGRUCell

                # Init new cell with trained weights
                self.rnn = AnalysableGRUCell(
                    num_layers=self.rnn.num_layers, batch_first=self.rnn.batch_first,
                    w_hh=self.rnn.weight_hh_l0, w_ih=self.rnn.weight_ih_l0, b_hh=self.rnn.bias_hh_l0,
                    b_ih=self.rnn.bias_ih_l0, save_dont_return=save_dont_return
                )

            self.cells_replaced = True


class AnalysableLSTMCell(nn.Module):
    def __init__(self, num_layers, w_ih, w_hh, b_ih=None, b_hh=None, batch_first=True, save_dont_return=False):
        super().__init__()
        self.bidirectional = False
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Give the option to save gate activations instead of returning them (useful when modifying the return values
        # of a function isn't an option)
        self.save_dont_return = save_dont_return
        self.gates = None

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

        output = hy[self.num_layers - 1].clone().unsqueeze(0)

        gates = {
            "input_gate_activations": ingate, "forget_gate_activations": forgetgate,
            "output_gate_activations": outgate
        }

        if self.save_dont_return:
            self.gates = gates
            return output, (hy, cy)
        else:
            return output, (hy, cy), gates


class AnalysableGRUCell(nn.Module):
    def __init__(self, num_layers, w_ih, w_hh, b_ih=None, b_hh=None, batch_first=True, save_dont_return=False):
        super().__init__()
        self.bidirectional = False
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Give the option to save gate activations instead of returning them (useful when modifying the return values
        # of a function isn't an option)
        self.save_dont_return = save_dont_return
        self.gates = None

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

        output = hy[self.num_layers - 1].clone().unsqueeze(0)

        # The pytorch implementation is different from the classical GRU cell, because it comprises a resetgate,
        # inputgate and newgate instead of update gate and reset gate (compare with
        # https://en.wikipedia.org/wiki/Gated_recurrent_unit).
        # Apparently, the weight matrices W_z, W_r and W_h have been combined into w_ih as well as U_z, U_r and U_h
        # into w_hh. That implies that the biases are also combined (b_z, b_r and b_h into b_ih AND b_hh), meaning
        # that a * b_ih + b * b_hh = concat(b_z, b_r, b_h) for a, b > 0 and a + b = 1, which doesn't change the results
        # for the reset and input (= update) gate.

        gates = {
            "input_gate_activations": inputgate, "reset_gate_activations": resetgate
        }

        if self.save_dont_return:
            self.gates = gates
            return output, hy
        else:
            return output, hy, gates
