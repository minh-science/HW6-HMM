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
        forward_probabilities = np.zeros( (len(input_observation_states), len(self.hidden_states))  )
       
        # Step 2. Calculate probabilities
        for i, obs_state in enumerate(input_observation_states): # number and go through observation states
            for j, hidden_state in enumerate(self.hidden_states): # number and go through hidden states 
                if i == 0:
                    forward_probabilities[i,j] = self.prior_p[j] * self.emission_p[j, self.observation_states_dict[obs_state]] # use prior to get emission probability
                else:
                    emission_component = self.emission_p[j, self.observation_states_dict[obs_state] ] # probabilty of emission
                    transition_component = self.transition_p[j, np.argmax(forward_probabilities[i-1])  ] # probabity of hidden state transition
                    ij_transition =  emission_component * transition_component # emission * transition 
                    #  recursion step, looks at the previous forward probabilities
                    fwd_gen = [ij_transition * forward_probabilities[i - 1, k ]  for k in range(len(self.hidden_states))] # uses prior prior state to calculate next probability 
                    forward_probabilities[i,j] = np.sum(  np.fromiter( fwd_gen, dtype=float) # sum of probabilities at the new end of trellis 
                    )

        # Step 3. Return final probability 
        return np.sum(forward_probabilities[-1, :]) # sum of probabilities at the end of the trellis 


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
        # store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
       
        # Step 2. Calculate Probabilities
        # Initialize variables
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        backpointers = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=int)

        # Initialization step
        viterbi_table[0] = self.prior_p * self.emission_p[:, self.observation_states_dict[decode_observation_states[0]]] # use prior to get emission probability for firsts

        # Recursion step
        for t in range(1, len(decode_observation_states)): # number and go through observation states
            for j, hidden_state in enumerate(self.hidden_states):  # number and go through hidden states 
                emission_prob = self.emission_p[j, self.observation_states_dict[decode_observation_states[t]]] # emission from observation states dictionary
                transition_probs = self.transition_p[:, j] * viterbi_table[t-1] # transition = self transition * previous 
                backpointers[t, j] = np.argmax(transition_probs) # pick most likely
                viterbi_table[t, j] = np.max(transition_probs) * emission_prob # update viterbi table 

        # Step 3. Traceback 
        best_hidden_states = [np.argmax(viterbi_table[-1])] # choose best from viterbi table
        for t in range(len(decode_observation_states) - 1, 0, -1):
            best_hidden_states.insert(0, backpointers[t, best_hidden_states[0]]) # get best hidden state

        # Step 4. Return best hidden state sequence 
        return [self.hidden_states[state] for state in best_hidden_states]