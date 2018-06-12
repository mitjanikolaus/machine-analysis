"""
Define versions of RNN cells which enable the user to record the activations.
"""

# EXT
import torch.nn.functional as F
from torch import nn


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
            'input_gate_activations': ingate, 'forget_gate_activations': forgetgate,
            'output_gate_activations': outgate, 'cell_gate_activations': cy_tilde
        }


# def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#
#     if input.is_cuda:
#         gi = F.linear(input, w_ih)
#         gh = F.linear(hidden, w_hh)
#         state = fusedBackend.GRUFused.apply
#         return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)
#
#     gi = F.linear(input, w_ih, b_ih)
#     gh = F.linear(hidden, w_hh, b_hh)
#     i_r, i_i, i_n = gi.chunk(3, 1)
#     h_r, h_i, h_n = gh.chunk(3, 1)
#
#     resetgate = F.sigmoid(i_r + h_r)
#     inputgate = F.sigmoid(i_i + h_i)
#     newgate = F.tanh(i_n + resetgate * h_n)
#     hy = newgate + inputgate * (hidden - newgate)
#
#     return hy