
# STD
import random
import numpy as np
from collections import defaultdict

# EXT
from torch import nn
import torch.nn.functional as F

# PROJECT
from seq2seq.models import DecoderRNN
from seq2seq.models.attention import HardGuidance
from models.analysable_cells import AnalysableLSTMCell, AnalysableGRUCell, AnalysableCellsMixin


class HiddenStateAnalysisDecoderRNN(DecoderRNN, AnalysableCellsMixin):
    KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS = 'hidden_activations_decoder'
    KEY_CELL_ACTIVATIONS_ALL_TIMESTEPS = 'cell_activations_decoder'

    def __init__(self, *args, **kwargs):
        DecoderRNN.__init__(self, *args, **kwargs)
        AnalysableCellsMixin.__init__(self, *args, **kwargs)

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0, provided_attention=None):
        hidden_activations_all_timesteps = []
        cell_activations_all_timesteps = []
        gate_activations_all_timesteps = defaultdict(list)

        return_dict = dict()
        if self.use_attention:
            return_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)

        # Replace cells once
        # Because we don't modify the original decoder's code, save the gate activations of the current time step inside
        # the object instead of returning them
        self.replace_cells(save_dont_return=True)

        decoder_hidden_states = self._init_state(encoder_hidden)
        if self.rnn_cell == AnalysableLSTMCell:
            decoder_cell_states = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                return_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Prepare extra arguments for attention method
        attention_method_kwargs = {}
        if self.attention and isinstance(self.attention.method, HardGuidance):
            attention_method_kwargs['provided_attention'] = provided_attention

        # Always unroll the decoder to be able to access the hidden activations
        symbols = None
        for di in range(max_length):
            # We always start with the SOS symbol as input. We need to add extra dimension of length 1 for the number of decoder steps (1 in this case)
            # When we use teacher forcing, we always use the target input.
            if di == 0 or use_teacher_forcing:
                decoder_input = inputs[:, di].unsqueeze(1)
            # If we don't use teacher forcing (and we are beyond the first SOS step), we use the last output as new input
            else:
                decoder_input = symbols

            # Perform one forward step
            if self.attention and isinstance(self.attention.method, HardGuidance):
                attention_method_kwargs['step'] = di

            if self.rnn_cell == AnalysableLSTMCell:
                decoder_output, (decoder_hidden_states, decoder_cell_states), step_attn = self.forward_step(
                    decoder_input, (decoder_hidden_states, decoder_cell_states), encoder_outputs,
                    function=function, **attention_method_kwargs
                )

                cell_activations_all_timesteps.append(decoder_cell_states)
            else:
                decoder_output, decoder_hidden_states, step_attn = self.forward_step(
                    decoder_input, decoder_hidden_states, encoder_outputs, function=function, **attention_method_kwargs
                )

            if self.rnn_cell in (AnalysableLSTMCell, AnalysableGRUCell):
                gates = self.rnn.gates
                for gate_name, activations in gates.items():
                    gate_activations_all_timesteps[gate_name].append(activations)

            hidden_activations_all_timesteps.append(decoder_hidden_states)

            # Remove the unnecessary dimension.
            step_output = decoder_output.squeeze(1)
            # Get the actual symbol
            symbols = decode(di, step_output, step_attn)

        return_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        return_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()
        return_dict[HiddenStateAnalysisDecoderRNN.KEY_HIDDEN_ACTIVATIONS_ALL_TIMESTEPS] = hidden_activations_all_timesteps

        if self.rnn_cell == AnalysableLSTMCell:
            return_dict[HiddenStateAnalysisDecoderRNN.KEY_CELL_ACTIVATIONS_ALL_TIMESTEPS] = cell_activations_all_timesteps

        for gate_name, all_activations in gate_activations_all_timesteps.items():
            return_dict[gate_name + "_decoder"] = all_activations

        return decoder_outputs, decoder_hidden_states, return_dict
