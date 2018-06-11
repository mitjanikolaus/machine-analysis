import os
import dill
from seq2seq.models import DecoderRNN
from seq2seq.models import EncoderRNN
from seq2seq.models import Seq2seq
from seq2seq.models.attention import MLP, Dot, HardGuidance, Concat
from seq2seq.util.checkpoint import Checkpoint
from torch.nn import GRU, LSTM

import torch.nn.functional as F


from .HiddenStateAnalysisDecoderRNN import HiddenStateAnalysisDecoderRNN

class AnalysableSeq2seq(Seq2seq):


    @staticmethod
    def load(path_to_checkpoint: str):


        checkpoint = Checkpoint.load(path_to_checkpoint)

        if isinstance(checkpoint.model.decoder.attention.method, MLP):
            attention_method = 'mlp'
        elif isinstance(checkpoint.model.decoder.attention.method, Dot):
            attention_method = 'dot'
        elif isinstance(checkpoint.model.decoder.attention.method, HardGuidance):
            attention_method = 'hard'
        elif isinstance(checkpoint.model.decoder.attention.method, Concat):
            attention_method = 'concat'
        else:
            raise ImportError('Model trained with unknown attention method, cannot be loaded')

        if isinstance(checkpoint.model.decoder.rnn, GRU):
            decoder_rnn_cell = 'gru'
        elif isinstance(checkpoint.model.decoder.rnn, LSTM):
            decoder_rnn_cell = 'lstm'
        else:
            raise ImportError('Model trained with unknown rnn cell type, cannot be loaded')

        if isinstance(checkpoint.model.decoder.rnn, GRU):
            encoder_rnn_cell = 'gru'
        elif isinstance(checkpoint.model.decoder.rnn, LSTM):
            encoder_rnn_cell = 'lstm'
        else:
            raise ImportError('Model trained with unknown rnn cell type, cannot be loaded')

        # TODO: override Decoder and Encoder to provide required stats
        decoder = HiddenStateAnalysisDecoderRNN(
            vocab_size=len(checkpoint.output_vocab),
            max_len=checkpoint.model.decoder.max_len,
            hidden_size=checkpoint.model.decoder.hidden_size,
            sos_id=checkpoint.output_vocab.stoi['<sos>'],
            eos_id=checkpoint.output_vocab.stoi['<eos>'],
            use_attention=checkpoint.model.decoder.use_attention,
            attention_method=attention_method,
            full_focus=checkpoint.model.decoder.full_focus,
            n_layers=checkpoint.model.decoder.n_layers,
            rnn_cell=decoder_rnn_cell,
            bidirectional=checkpoint.model.decoder.rnn.bidirectional,
            input_dropout_p=checkpoint.model.decoder.input_dropout_p,
            dropout_p=checkpoint.model.decoder.dropout_p,
        )
        encoder = EncoderRNN(
            vocab_size=len(checkpoint.input_vocab),
            max_len=checkpoint.model.encoder.max_len,
            hidden_size=checkpoint.model.encoder.hidden_size,
            embedding_size=checkpoint.model.encoder.embedding_size,
            input_dropout_p=checkpoint.model.encoder.input_dropout_p,
            dropout_p=checkpoint.model.encoder.dropout_p,
            n_layers=checkpoint.model.encoder.n_layers,
            bidirectional=checkpoint.model.encoder.rnn.bidirectional,
            rnn_cell=encoder_rnn_cell,
            variable_lengths=checkpoint.model.encoder.variable_lengths
        )

        analysableSeq2seq = AnalysableSeq2seq(encoder, decoder)
        analysableSeq2seq.load_state_dict(checkpoint.model.state_dict())

        model = analysableSeq2seq

        model.flatten_parameters()  # make RNN parameters contiguous

        return Checkpoint(model=model, input_vocab=checkpoint.input_vocab,
                          output_vocab=checkpoint.output_vocab,
                          optimizer=checkpoint.optimizer,
                          epoch=checkpoint.epoch,
                          step=checkpoint.step,
                          path=path_to_checkpoint)


#model = AnalysableSeq2seq.load('../../machine-zoo/guided/gru/1')
#print(model)
