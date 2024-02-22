import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    observation_state_sequence = mini_input["observation_state_sequence"]
    best_hidden_state_sequence = mini_input["best_hidden_state_sequence"]

    mini_hmm_forward = HiddenMarkovModel(
        observation_states= mini_hmm["observation_states"],
        hidden_states=mini_hmm["hidden_states"], 
        prior_p=mini_hmm["prior_p"],
        transition_p=mini_hmm["transition_p"],
        emission_p = mini_hmm["emission_p"]
    )
    mini_hmm_forward.forward(input_observation_states= observation_state_sequence )
    pass

test_mini_weather()


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    pass













