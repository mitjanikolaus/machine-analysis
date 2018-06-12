"""
Define GRU and LSTM cells that store activations for different gates.
"""

# EXT
import torch.functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

class AnalysableLSTMCell:
    def __init__(self):
        self.ingate = None
        self.forgetgate = None
        self.cellgate = None
        self.outgate = None

    @property
    def current_gate_activations(self):
        return {"in": self.ingate, "forget": self.forgetgate, "cell": self.cellgate, "out": self.outgate}

    def __call__(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        print("Analyzable cell is used")

        if input.is_cuda:
            igates = F.linear(input, w_ih)
            hgates = F.linear(hidden[0], w_hh)
            state = fusedBackend.LSTMFused.apply
            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        self.ingate = F.sigmoid(ingate)
        self.forgetgate = F.sigmoid(forgetgate)
        self.cellgate = F.tanh(cellgate)
        self.outgate = F.sigmoid(outgate)

        cy = (self.forgetgate * cx) + (self.ingate * self.cellgate)
        hy = self.outgate * F.tanh(cy)

        return hy, cy


class AnalysableGRUCell:

    def __init__(self):
        self.resetgate = None
        self.inputgate = None
        self.newgate = None

    @property
    def current_gate_activations(self):
        return {"reset": self.resetgate, "input": self.inputgate, "new": self.newgate}

    def __call__(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        if input.is_cuda:
            gi = F.linear(input, w_ih)
            gh = F.linear(hidden, w_hh)
            state = fusedBackend.GRUFused.apply
            return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)

        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        self.resetgate = F.sigmoid(i_r + h_r)
        self.inputgate = F.sigmoid(i_i + h_i)
        self.newgate = F.tanh(i_n + self.resetgate * h_n)
        hy = self.newgate + self.inputgate * (hidden - self.newgate)

        return hy

