
import numpy as np
import matplotlib.pyplot as plt

import torchtext

from models.analysable_seq2seq import AnalysableSeq2seq

from activations import ActivationsDataset


def run_analysis(rnn_type, similar_input_criterium, other_input_criterium, models_to_evaluate, model_part_to_evaluate, compare, print_outliers=True):

    distances_baseline_similar_all_models = []
    distances_baseline_other_all_models = []
    distances_guided_similar_all_models = []
    distances_guided_other_all_models = []

    for model_id in models_to_evaluate:
        for i, model_type in enumerate(['baseline', 'guided']):
            print('evaluating ',model_type, model_id)

            activations_dataset_path = 'data/' + model_type + '_' + rnn_type + '_'+str(model_id)+'_all.pt'

            samples_group_similar, samples_group_other = prepare_test_data(activations_dataset_path, similar_input_criterium, other_input_criterium, model_part_to_evaluate)

            distances_similar, distances_different = run_models_and_evaluate(samples_group_similar, samples_group_other, compare, print_outliers)

            if(model_type == 'baseline'):
                distances_baseline_similar_all_models.append(distances_similar)
                distances_baseline_other_all_models.append(distances_different)
            else:
                distances_guided_similar_all_models.append(distances_similar)
                distances_guided_other_all_models.append(distances_different)

    distances_baseline_similar_all_models = np.mean(distances_baseline_similar_all_models, axis=0)
    distances_baseline_other_all_models = np.mean(distances_baseline_other_all_models, axis=0)
    distances_guided_similar_all_models = np.mean(distances_guided_similar_all_models, axis=0)
    distances_guided_other_all_models = np.mean(distances_guided_other_all_models, axis=0)

    plt.subplot(2, 1, 1)
    plt.title('Baseline model')
    plt.plot(np.arange(len(distances_baseline_similar_all_models)), distances_baseline_similar_all_models, label="similar inputs", linestyle='-',
             marker="o")
    plt.plot(np.arange(len(distances_baseline_other_all_models)), distances_baseline_other_all_models, label="baseline inputs", linestyle='-',
             marker="o")
    if compare == COMPARE_SAMPLES:
        plt.xticks(np.arange(len(distances_baseline_other_all_models)), ['t={}'.format(i) for i in np.arange(len(distances_baseline_other_all_models))])
    if compare == COMPARE_TIMESTEPS:
        plt.xticks(np.arange(len(distances_baseline_other_all_models)), ['t{} -> t{}'.format(i, i+1) for i in np.arange(len(distances_baseline_other_all_models))])

    plt.ylabel('Mean distances')
    plt.legend(loc=2)

    plt.subplot(2, 1, 2)
    plt.title('Guided Attention Model')
    plt.plot(np.arange(len(distances_guided_similar_all_models)), distances_guided_similar_all_models,
             label="similar inputs", linestyle='-',
             marker="o")
    plt.plot(np.arange(len(distances_guided_other_all_models)), distances_guided_other_all_models,
             label="baseline inputs", linestyle='-',
             marker="o")
    if compare == COMPARE_SAMPLES:
        plt.xticks(np.arange(len(distances_baseline_other_all_models)), ['t={}'.format(i) for i in np.arange(len(distances_baseline_other_all_models))])
    if compare == COMPARE_TIMESTEPS:
        plt.xticks(np.arange(len(distances_baseline_other_all_models)), ['t{} -> t{}'.format(i, i+1) for i in np.arange(len(distances_baseline_other_all_models))])

    plt.ylabel('Mean distances')
    plt.legend(loc=2)

    plt.show()

def run_models_and_evaluate(activation_data_similar, activation_data_other, compare, print_outliers=True):
    if compare == COMPARE_SAMPLES:
        distances_similar_each_timestep = calculate_distances_among_samples(activation_data_similar)
        distances_different_each_timestep = calculate_distances_among_samples(activation_data_other)
    elif compare == COMPARE_TIMESTEPS:
        distances_similar_each_timestep = calculate_distances_among_timesteps(activation_data_similar)
        distances_different_each_timestep = calculate_distances_among_timesteps(activation_data_other)
    else:
        raise ValueError("compare must be either COMPARE_TIMESTEPS or COMPARE_SAMPLES")

    means_similar_each_timestep = []
    means_different_each_timestep = []

    for ts in range(len(distances_similar_each_timestep)):
        mean_distances_similar_per_cell = np.mean(distances_similar_each_timestep[ts], axis=0)
        mean_distances_different_per_cell = np.mean(distances_different_each_timestep[ts], axis=0)

        if print_outliers:
            outliers_similar = np.where(mean_distances_similar_per_cell < (np.mean(mean_distances_similar_per_cell) - 2 * np.std(mean_distances_similar_per_cell)))[0]
            outliers_different = np.where(mean_distances_different_per_cell < (np.mean(mean_distances_different_per_cell) - 2 * np.std(mean_distances_different_per_cell)))[0]
            relevant_outliers = list(set(outliers_similar) - set(outliers_different))
            relevant_outliers.sort()
            print('Relevant outliers timestep', ts, relevant_outliers)

        means_similar_each_timestep.append(np.sqrt(np.sum(mean_distances_similar_per_cell)))
        means_different_each_timestep.append(np.sqrt(np.sum(mean_distances_different_per_cell)))

    return means_similar_each_timestep, means_different_each_timestep

def calculate_distances_among_samples(activation_data):
        distances = []
        for i, input_sample in enumerate(activation_data):
            for ts, input_sample_timestep in enumerate(input_sample):
                if len(distances) <= ts:
                    distances.append([])
                for j, input_sample_2 in enumerate(activation_data):
                    if not j == i:
                        if len(input_sample_2) >= len(input_sample):
                            single_distances = (np.square(np.subtract(input_sample_timestep, input_sample_2[ts])))
                            distances[ts].append(single_distances)
        return distances

def calculate_distances_among_timesteps(activation_data):
    distances = []
    for i, input_sample in enumerate(activation_data):
        for ts, input_sample_timestep in enumerate(input_sample):
            if ts < len(input_sample)-1:
                if len(distances) <= ts:
                    distances.append([])
                single_distances = (np.square(np.subtract(input_sample_timestep, input_sample[ts+1])))
                distances[ts].append(single_distances)
    return distances

def prepare_test_data(activations_dataset_path, similar_input_criterium, other_input_criterium, model_part_to_evaluate):
    samples_group_similar = []  # all samples that start with t1
    samples_group_other = []  # all other samples

    activation_dataset = ActivationsDataset.load(activations_dataset_path, convert_to_numpy=True)

    for input, activation_data in zip(activation_dataset.model_inputs, getattr(activation_dataset, model_part_to_evaluate)):
        if similar_input_criterium(input[0]):
            samples_group_similar.append(activation_data)
        if other_input_criterium(input[0]):
            samples_group_other.append(activation_data)

    return samples_group_similar, samples_group_other

#Hidden units
ACTIVATIONS_HIDDEN_UNITS_DECODER = 'hidden_activations_decoder'
ACTIVATIONS_HIDDEN_UNITS_ENCODER = 'hidden_activations_encoder'

#GRU Gates
ACTIVATIONS_GRU_INPUT_GATE_DECODER = 'input_gate_activations_decoder'
ACTIVATIONS_GRU_INPUT_GATE_ENCODER = 'input_gate_activations_encoder'
ACTIVATIONS_GRU_NEW_GATE_DECODER = 'new_gate_activations_decoder'
ACTIVATIONS_GRU_NEW_GATE_ENCODER = 'new_gate_activations_encoder'
ACTIVATIONS_GRU_RESET_GATE_DECODER = 'reset_gate_activations_decoder'
ACTIVATIONS_GRU_RESET_GATE_ENCODER = 'reset_gate_activations_encoder'

#LSTM Gates
ACTIVATIONS_LSTM_CELL_GATE_DECODER = 'cell_gate_activations_decoder'
ACTIVATIONS_LSTM_CELL_GATE_ENCODER = 'cell_gate_activations_encoder'
ACTIVATIONS_LSTM_INPUT_GATE_DECODER = 'input_gate_activations_decoder'
ACTIVATIONS_LSTM_INPUT_GATE_ENCODER = 'input_gate_activations_encoder'
ACTIVATIONS_LSTM_FORGET_GATE_DECODER = 'forget_gate_activations_decoder'
ACTIVATIONS_LSTM_FORGET_GATE_ENCODER = 'forget_gate_activations_encoder'
ACTIVATIONS_LSTM_OUTPUT_GATE_DECODER = 'output_gate_activations_decoder'
ACTIVATIONS_LSTM_OUTPUT_GATE_ENCDODER = 'output_gate_activations_encoder'

COMPARE_SAMPLES = 'compare_among_different_samples'
COMPARE_TIMESTEPS = 'compare_among_different_timesteps'


if __name__ == "__main__":
    models_to_evaluate = [1,2,3,4,5]  # 1,2,3,4,5 (which models from the zoo to take)
    print_outliers = True

    rnn_type = 'lstm' #gru or lstm

    model_part_to_evaluate = ACTIVATIONS_LSTM_OUTPUT_GATE_DECODER

    # either compare among different samples or compare among different timesteps
    compare = COMPARE_TIMESTEPS # COMPARE_SAMPLES or COMPARE_TIMESTEPS

    # load a checkpoint to get input vocab
    checkpoint_path = '../machine-zoo/baseline/' + rnn_type + '/' + str(models_to_evaluate[0]) + '/'
    checkpoint = AnalysableSeq2seq.load(checkpoint_path)
    input_vocab = checkpoint.input_vocab

    def similar_input_criterium(sample):
        return  input_vocab.itos[sample[1]] == 't2' and input_vocab.itos[sample[2]] == 't2' #input_vocab.itos[sample[0]] == '000'

    #return just True to get a baseline of all samples
    def other_input_criterium(sample):
        return True #and input_vocab.itos[sample[0]] == '000'

    run_analysis(rnn_type, similar_input_criterium, other_input_criterium, models_to_evaluate, model_part_to_evaluate, compare, print_outliers=print_outliers)