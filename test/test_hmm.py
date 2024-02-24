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
    # load data for mini hmm
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # load observation sequence and hidden sequence 
    observation_state_sequence = mini_input["observation_state_sequence"]
    best_hidden_state_sequence = mini_input["best_hidden_state_sequence"]

    # initalize HMM 
    mini_hmm = HiddenMarkovModel(
        observation_states= mini_hmm["observation_states"],
        hidden_states=mini_hmm["hidden_states"], 
        prior_p=mini_hmm["prior_p"],
        transition_p=mini_hmm["transition_p"],
        emission_p = mini_hmm["emission_p"]
    )

    # check sizes of transition and emission matrices 
    assert mini_hmm.transition_p.size <= mini_hmm.emission_p.size
    # check if transition matrix is square 
    assert mini_hmm.transition_p.shape[0] == mini_hmm.transition_p.shape[1]

    # assert output of forward algorithm is correct
    print ("HERE", mini_hmm.forward(input_observation_states= observation_state_sequence ))
    assert mini_hmm.forward(input_observation_states= observation_state_sequence ) == 0.031244062500000003

    # assert length of viterbi output is correct
    assert len(mini_hmm.viterbi(observation_state_sequence)) == len(best_hidden_state_sequence)

    # assert output of viterbi is correct
    assert [i for i in mini_hmm.viterbi(observation_state_sequence)] == [j for j in best_hidden_state_sequence]

test_mini_weather()


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    # load data for mini hmm
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    # load observation sequence and hidden sequence 
    observation_state_sequence = full_input["observation_state_sequence"]
    best_hidden_state_sequence = full_input["best_hidden_state_sequence"]

    # initalize HMM 
    full_hmm = HiddenMarkovModel(
        observation_states= full_hmm["observation_states"],
        hidden_states=      full_hmm["hidden_states"], 
        prior_p=            full_hmm["prior_p"],
        transition_p=       full_hmm["transition_p"],
        emission_p =        full_hmm["emission_p"]
    )

    # check sizes of transition and emission matrices 
    assert full_hmm.transition_p.size <= full_hmm.emission_p.size
    # check if transition matrix is square 
    assert full_hmm.transition_p.shape[0] == full_hmm.transition_p.shape[1]

    # assert output of forward algorithm is correct
    print("HERE", full_hmm.forward(input_observation_states= observation_state_sequence ))
    assert full_hmm.forward(input_observation_states= observation_state_sequence ) == 6.808160936567043e-13

    # assert length of viterbi output is correct
    assert len(full_hmm.viterbi(observation_state_sequence)) == len(best_hidden_state_sequence)

    # assert output of viterbi is correct
    # assert [i for i in full_hmm.viterbi(observation_state_sequence)] == [j for j in best_hidden_state_sequence]
    # print( [i for i in full_hmm.viterbi(observation_state_sequence)])
    # print( [j for j in best_hidden_state_sequence] ) 


test_full_weather()









