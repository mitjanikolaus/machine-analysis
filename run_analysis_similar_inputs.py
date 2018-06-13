
import numpy as np
import matplotlib.pyplot as plt

import torchtext

from models.analysable_seq2seq import AnalysableSeq2seq
from pygments.lexer import include

from util import load_test_data, plot_activation_distributions
from get_hidden_activations import run_and_get_hidden_activations_with_test_set
from activations import ActivationsDataset

def run_analysis():
    ignore_output_eos = True
    use_attention_loss = True
    attention_method = 'mlp'

    rnn_type = 'gru'   # gru or lstm
    model_part_to_evaluate = 'decoder'  # encoder or decoder
    model_id = 1    # 1,2,3,4,5 (which model from the zoo to take)
    timesteps_to_evaluate = [0,1,2]

    def similar_input_criterium(sample):
        return len(sample.src) == 4 and sample.src[1] == 't1' and sample.src[0] == '001'

    def other_input_criterium(sample):
        return len(sample.src) == 4 and sample.src[0] == '001'

    test_data_1 = '../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/heldout_compositions.tsv'
    test_data_2 = '../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/heldout_inputs.tsv'
    test_data_3 = '../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/heldout_tables.tsv'
    test_data_4 = '../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/new_compositions.tsv'
    test_data_5 = '../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/train.tsv'
    test_data_paths = [test_data_1, test_data_2, test_data_3, test_data_4, test_data_5]

    distances_similar_baseline = []
    distances_different_baseline = []
    distances_similar_guided = []
    distances_different_guided = []

    for timestep in timesteps_to_evaluate:
        (mean_distance_similar_baseline, mean_distance_different_baseline) = run_model_and_evaluate('baseline', rnn_type, test_data_paths, timestep, model_part_to_evaluate, model_id, similar_input_criterium, other_input_criterium, ignore_output_eos, use_attention_loss, attention_method)
        (mean_distance_similar_guided, mean_distance_different_guided) = run_model_and_evaluate('guided', rnn_type, test_data_paths, timestep, model_part_to_evaluate, model_id, similar_input_criterium, other_input_criterium, ignore_output_eos, use_attention_loss, attention_method)

        distances_similar_baseline.append(mean_distance_similar_baseline)
        distances_different_baseline.append(mean_distance_different_baseline)
        distances_similar_guided.append(mean_distance_similar_guided)
        distances_different_guided.append(mean_distance_different_guided)
        print('\nTimestep: ',timestep)
        print('Baseline:\n Similar inputs: ',mean_distance_similar_baseline,' Different inputs: ',mean_distance_different_baseline, 'Delta: '+ str(mean_distance_similar_baseline-mean_distance_different_baseline))
        print('Guided:\n Similar inputs: ',mean_distance_similar_guided,' Different inputs: ',mean_distance_different_guided, 'Delta: '+ str(mean_distance_similar_guided-mean_distance_different_guided))

    plt.subplot(2, 1, 1)
    plt.plot(timesteps_to_evaluate, distances_similar_baseline, label="similar_inputs")
    plt.plot(timesteps_to_evaluate, distances_different_baseline, label="different_inputs")
    plt.xticks(np.arange(min(timesteps_to_evaluate), max(timesteps_to_evaluate), step=1.0))
    #y_max = np.amax(np.array([distances_similar_baseline, distances_different_baseline, distances_similar_guided, distances_different_guided]))
    #y_min = np.amin(np.array([distances_similar_baseline, distances_different_baseline, distances_similar_guided, distances_different_guided]))
    #plt.yticks(np.arange(y_min, y_max))
    plt.ylabel('Mean distance baseline')
    plt.legend(loc=2)

    plt.subplot(2, 1, 2)
    plt.plot(timesteps_to_evaluate, distances_similar_guided, label="similar_inputs")
    plt.plot(timesteps_to_evaluate, distances_different_guided, label="different_inputs")
    plt.xticks(np.arange(min(timesteps_to_evaluate), max(timesteps_to_evaluate), step=1.0))
    #plt.yticks(np.arange(y_min, y_max))
    plt.xlabel('timestep')
    plt.ylabel('Mean distance guided')
    plt.legend(loc=2)

    plt.show()

def run_model_and_evaluate(model_type, rnn_type, test_data_paths, timestep_to_evaluate, model_part_to_evaluate, model_id, similar_input_criterium, other_input_criterium, ignore_output_eos, use_attention_loss, attention_method):
    activations_path_similar = 'activations_similar_'+model_type+'_' + rnn_type + '.pt'
    activations_path_different = 'activations_different_'+model_type+'_' + rnn_type + '.pt'

    checkpoint_path = '../machine-zoo/'+model_type+'/' + rnn_type + '/'+str(model_id)+'/'

    checkpoint = AnalysableSeq2seq.load(checkpoint_path)
    model = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    (test_set_similar, test_set_different) = prepare_test_data(test_data_paths, input_vocab, output_vocab, similar_input_criterium, other_input_criterium, ignore_output_eos, use_attention_loss, attention_method)

    run_and_get_hidden_activations_with_test_set(model, test_set_similar, save_path=activations_path_similar)
    run_and_get_hidden_activations_with_test_set(model, test_set_different, save_path=activations_path_different)

    activation_data_similar = ActivationsDataset.load(activations_path_similar)
    activation_data_different = ActivationsDataset.load(activations_path_different)

    result = ()
    for data in (activation_data_similar, activation_data_different):
        distances = []
        for input_sample in getattr(data,'hidden_activations_' + model_part_to_evaluate):
            sample = input_sample[timestep_to_evaluate].numpy().flatten()
            for input_sample_2 in getattr(data,'hidden_activations_' + model_part_to_evaluate):
                sample2 = input_sample_2[timestep_to_evaluate].numpy().flatten()
                distances.append(euclidean_distance(sample, sample2))

        result = result + (np.mean(distances),)

    return result


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def prepare_test_data(test_data_paths, input_vocab, output_vocab, similar_input_criterium, other_input_criterium, ignore_output_eos, use_attention_loss, attention_method='mlp', max_len=50):

    samples_group_similar = []  # all samples that start with t1
    samples_group_other = []  # all other samples

    for test_data_path in test_data_paths:
        test_set = load_test_data(test_data_path, input_vocab, output_vocab, ignore_output_eos, use_attention_loss,
                              attention_method, max_len)

        for sample in test_set.examples:
            if similar_input_criterium(sample):
                samples_group_similar.append(sample)
            if other_input_criterium(sample):
                samples_group_other.append(sample)

    test_set_similar = torchtext.data.Dataset(
        examples=samples_group_similar,
        fields=test_set.fields,
        filter_pred = None
    )

    test_set_different = torchtext.data.Dataset(
        examples=samples_group_other,
        fields=test_set.fields,
        filter_pred = None
    )

    print('Similar samples: ', [' '.join(item.src) for item in test_set_similar])
    print('Other samples: ', [' '.join(item.src) for item in test_set_different])

    return (test_set_similar, test_set_different)




run_analysis()