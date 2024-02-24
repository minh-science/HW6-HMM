import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        print("observation states:", input_observation_states)
        print("observation states dict:", self.observation_states_dict)
        print("hidden states:", self.hidden_states)
        print("transition matrix", self.transition_p)
        print("emission matrix", self.emission_p)
        print("hidden dict", self.hidden_states_dict)
        forward_probabilities = np.zeros( (len(input_observation_states), len(self.hidden_states))  )
        # print((len(input_observation_states), len(self.hidden_states)))
        # print(len(input_observation_states))
        # print(len(self.observation_states))
        # print(self.observation_states_dict)
        # print(self.emission_p)
       
        # Step 2. Calculate probabilities
        for i, obs_state in enumerate(input_observation_states): # number and go through observation states
            for j, hidden_state in enumerate(self.hidden_states): # number and go through hidden states 
                if i == 0:
                    forward_probabilities[i,j] = self.prior_p[j] * self.emission_p[j, self.observation_states_dict[obs_state]] # use prior to get emission probability
                else:
                    # forward_probabilities[t, j] = np.sum(forward_probabilities[t - 1, i] * self.transition_p[i, j] * self.emission_p[j, self.observation_states_dict[obs]] for i in range(len(self.hidden_states)))
                    print(j, obs_state, self.observation_states_dict[obs_state] )
                    emission_component = self.emission_p[j, self.observation_states_dict[obs_state] ] 

                    # transition_component = self.transition_p[j, self.observation_states_dict[obs_state]]
                    
                    ij_transition =  emission_component 
                    #  recursion step, looks at the previousforward probabilities
                    fwd_gen = [ij_transition * forward_probabilities[i - 1, k ]  for k in range(len(self.hidden_states))]
                    forward_probabilities[i,j] = np.sum(  np.fromiter(
                        # forward_probabilities[i - 1, k ] * self.transition_p[j, self.observation_states_dict[obs_state]] for k in range(len(self.hidden_states))
                        fwd_gen, dtype=float
                        )
                    )
            # print(forward_probabilities) 
            # print(self.hidden_states)
            print(forward_probabilities[i], np.max(forward_probabilities[i]), self.hidden_states[np.argmax(forward_probabilities[i])])
        print(forward_probabilities)
        # Step- 
        fwd_hmm_seq = []
        for row in forward_probabilities: # transpose matrix, get columns of original matrix (rows of transpose matrix)
            print(row, np.max(row), self.hidden_states[np.argmax(row)])
            fwd_hmm_seq.append(self.hidden_states[np.argmax(row)])
            pass
        print(fwd_hmm_seq)


        # Step 3. Return final probability 
        print("transition matrix:", transition_p)
        return np.sum(forward_probabilities[-1, :])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))   


        
       
        # Step 2. Calculate Probabilities
        for i, obs_state in enumerate(decode_observation_states):  
            print(f"observation {i}:", obs_state)
            for j, hidden_state in enumerate(self.hidden_states):
                print("this is the emission", self.emission_p[j])
                # print("hidden state:",j,hidden_state)
                if i == 0: # initialize viterbi state 0 
                    print("this is the prior", self.prior_p)
                    # print(i,j)
                    # print(self.emission_p[i,j])
                    viterbi_table[i,j] = self.prior_p[i] * self.emission_p[i,j]
                    print(self.emission_p)
        print(viterbi_table)



        for i, obs_state in enumerate(decode_observation_states): # number and go through observation states
            for j, hidden_state in enumerate(self.hidden_states): # number and go through hidden states 
                if i == 0:
                    viterbi_table[i,j] = self.prior_p[j] * self.emission_p[j, self.observation_states_dict[obs_state]] # use prior to get emission probability
                else:
                    # forward_probabilities[t, j] = np.sum(forward_probabilities[t - 1, i] * self.transition_p[i, j] * self.emission_p[j, self.observation_states_dict[obs]] for i in range(len(self.hidden_states)))
                    print(j, obs_state, self.observation_states_dict[obs_state] )
                    emission_component = self.emission_p[j, self.observation_states_dict[obs_state] ] 

                    # transition_component = self.transition_p[j, self.observation_states_dict[obs_state]]
                    
                    ij_transition =  emission_component 
                    #  recursion step, looks at the previousforward probabilities
                    fwd_gen = [ij_transition * viterbi_table[i - 1, k ]  for k in range(len(self.hidden_states))]
                    viterbi_table[i,j] = np.sum(  np.fromiter(
                        # forward_probabilities[i - 1, k ] * self.transition_p[j, self.observation_states_dict[obs_state]] for k in range(len(self.hidden_states))
                        fwd_gen, dtype=float
                        )
                    )
            # print(forward_probabilities) 
            # print(self.hidden_states)
            print(viterbi_table[i], np.max(viterbi_table[i]), self.hidden_states[np.argmax(viterbi_table[i])])
        print(viterbi_table)
        # Step- 

        # Step 3. Traceback 
        fwd_hmm_seq = []
        for row in viterbi_table: # transpose matrix, get columns of original matrix (rows of transpose matrix)
            print(row, np.max(row), self.hidden_states[np.argmax(row)])
            fwd_hmm_seq.append(self.hidden_states[np.argmax(row)])
            pass
        print(fwd_hmm_seq)

        # Step 4. Return best hidden state sequence 
        return fwd_hmm_seq

# viterbi 

observation_state_sequence = ["1", "1", "0", "0", "1", "0", "1", "1", "1", "1", "1"],
transition_p = [[0.5, 0.5],
                [0.5, 0.5]]
emission_p = [[0.5, 0.5],
              [0.5, 0.5]]
# print(coin.viterbi(decode_observation_states= observation_state_sequence ) )

# mini dataset testing 

mini_data=np.load('./data/mini_weather_hmm.npz')
mini_input=np.load('./data/mini_weather_sequences.npz')
observation_state_sequence = mini_input["observation_state_sequence"]
best_hidden_state_sequence = mini_input["best_hidden_state_sequence"]
mini_hmm = HiddenMarkovModel(
    observation_states= mini_data["observation_states"],
    hidden_states=mini_data["hidden_states"], 
    prior_p=mini_data["prior_p"],
    transition_p=mini_data["transition_p"],
    emission_p = mini_data["emission_p"]
    )
# print("mini hmm obs states", mini_data["hidden_states"] ) 

# print(mini_hmm.forward(input_observation_states= observation_state_sequence ) )

print(mini_hmm.viterbi(decode_observation_states= observation_state_sequence ) )
# print("best hidden state sequence:", best_hidden_state_sequence)



# Full dataset testing

full_hmm=np.load('./data/full_weather_hmm.npz')
full_input=np.load('./data/full_weather_sequences.npz')
observation_state_sequence = full_input["observation_state_sequence"]
best_hidden_state_sequence = full_input["best_hidden_state_sequence"]
full_hmm_forward = HiddenMarkovModel(
    observation_states= full_hmm["observation_states"],
    hidden_states=      full_hmm["hidden_states"], 
    prior_p=            full_hmm["prior_p"],
    transition_p=       full_hmm["transition_p"],
    emission_p =        full_hmm["emission_p"]
    )
# print("best hidden state sequence:", best_hidden_state_sequence)
# print(full_hmm_forward.forward(input_observation_states= observation_state_sequence ) )
# print("best hidden state sequence:", best_hidden_state_sequence)
