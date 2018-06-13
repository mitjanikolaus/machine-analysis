from seq2seq.models import Seq2seq
from seq2seq.models.attention import MLP, Dot, HardGuidance, Concat
from seq2seq.util.checkpoint import Checkpoint
from torch.nn import GRU, LSTM

from .HiddenStateAnalysisDecoderRNN import HiddenStateAnalysisDecoderRNN
from .HiddenStateAnalysisEncoderRNN import HiddenStateAnalysisEncoderRNN

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
        encoder = HiddenStateAnalysisEncoderRNN(
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

    def forward(self, input_variable, input_lengths=None, target_variables=None,
                teacher_forcing_ratio=0):
        # Unpack target variables
        try:
            target_output = target_variables.get('decoder_output', None)
            # The attention target is preprended with an extra SOS step. We must remove this
            provided_attention = target_variables['attention_target'][:,1:] if 'attention_target' in target_variables else None
        except AttributeError:
            target_output = None
            provided_attention = None


        encoder_outputs, encoder_hidden, ret_dict_encoder = self.encoder(input_variable, input_lengths)
        result = self.decoder(inputs=target_output,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              provided_attention=provided_attention)
        return result + (ret_dict_encoder,)

#model = AnalysableSeq2seq.load('../../machine-zoo/guided/gru/1')
#print(model)
