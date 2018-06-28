# machine-analysis

This project was conducted as a Project AI for the Artificial Intelligence Master's programme of the Universiteit van 
Amsterdam in 2018. The goal was to analyze the activations of the
Attentive Guidance model proposed by [Hupkes et al. (2018)](https://arxiv.org/abs/1805.09657) and recover differences 
with respect to the baseline model, which could help understand the differences and bias future models in a way that
produces the same results, but renders the training procedure from the above paper unnecessary. 

Models were trained on [the table lookup task](https://github.com/mitjanikolaus/machine-tasks) and stored in a [model zoo](https://github.com/Kaleidophon/machine-zoo). 
The original models, coming from [here](https://github.com/i-machine-think/machine), where then augmented in order to allow the extraction
of activation values easily.

## Results
(Summary)

Our experiments highlighted crucial differences in the way neurons in the AG model respond to inputs for both the encoder 
and decoder in comparison to the baseline. We showed that neurons in the AG model display a very disparate distribution 
in activation values. Moreover, these values changed more between time steps, indicating a higher sensitivity to inputs 
that the baseline does not seem to possess. The fact that the gates of the AG model were more often saturated than the 
baseline ones is in accordance with this hypothesis. Further results pointed into the same direction, as we were able to 
identify groups of neurons that respond to specific input tokens that are smaller than for the baseline, possibly 
indicating a higher degree of specialization. That this behaviour does indeed lead the AG model to construct 
compositional solutions. We did not manage to say with certainty whether the AG model is better at encoding more general 
information about the input, like the length of the sequence.

## Requirements

All required python packages can be installed by running
    
    pip3 install -r requirements.txt
    
For PyTorch, a manual installation might be necessary. See the [official site](https://pytorch.org/)
for more information.

## Modules

This repository contains the following modules:

| Module name | Content | Used for report sections | 
| -----------:|:------- |:------------------------ | 
| ``activations.py`` | Store model activations in a special data set class | All |
| ``baseline_guided_classification.py`` | | |
| ``count_model_inspection.py`` | | |
| ``count_prediction.py`` | | |
| ``distributions.py`` | Quantify the distributions of activation  values and the change between time steps | 4.1, Appendix C |
| ``functional_groups.py`` | Learn a diagnostic classifier in order to learn functional groups | 4.5 |
| ``get_hidden_activations.py`` | Extract the activations produced by an model and store them in a special data set | All |
| ``inspect_gate_activations.py`` | Produce gate saturation plots | 4.2 |
| ``plots.py`` | | |
| ``similar_activations.py`` | Show how similar activations for similar / dissimilar samples are | 4.4, Appendix A |
| ``visualization.py`` | Create a variety of plots. | 4.1, Appendix C |  
| ``models.analysable_cells.py`` | Create GRU and LSTM that allow the retrieval of their activations | All |
| ``models.analysable_decoder.py`` | Create an open ``machine-task`` decoder | All |
| ``models.analysable_encoder.py`` | Create an open ``machine-task`` encoder | All |
| ``models.analysable_seq2seq.py`` | Create a ``machine-task`` model that stores all activations | All |
 
