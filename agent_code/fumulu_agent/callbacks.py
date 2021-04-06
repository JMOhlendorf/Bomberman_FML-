import os
import pickle
import random

import numpy as np
from scipy.special import softmax


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        #self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        
        ### ADDED ###

        #self.logger.info("Loading model from saved state in training mode.")
        #with open("beta.pt", "rb") as file:
        #    self.beta = pickle.load(file)


        ###   ###
        weights = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0]) # move and don't wait
        self.model = weights / weights.sum()

    else:
        self.logger.info("Loading model from saved state.")
        with open("beta.pt", "rb") as file:
            #self.model = pickle.load(file)
            self.beta = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # Use given code of no betas exist yet

    if not os.path.isfile("beta.pt"):
        
        ### GIVEN CODE:

        
        # todo Exploration vs exploitation
        random_prob = .3
        if self.train and random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])

        self.logger.debug("Querying model for action.")
        return np.random.choice(ACTIONS, p=self.model)
    
    else:
    
        ### ADDED ###
        
        # Exploration vs exploitation
        random_prob = .5
        if self.train and random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 96%: walk in any direction. 4% wait.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])

        self.logger.debug("Querying model for action.")

        Q = np.zeros(len(ACTIONS)-1) # neglect bomb for now
        for k in range(len(ACTIONS)-1):
            Q[k] = state_to_features(self, game_state) @ self.beta[k]
        
        p_actions = np.append(softmax(Q),0) # Use softmax to decide (NOT greedy argmax(Q)) and add bomb=0
        p_actions = p_actions / p_actions.sum() # normalize

        return np.random.choice(ACTIONS, p=p_actions) # act according to Q probabilities
        #return ACTIONS[np.argmax(Q)] # do action with highest return
        ###   ###
           
    


###### The output size of this function needs to be inserted in train.py when changing it
def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)

    ### ADDED ###

    # FIRST IDEA: features: empty inside arena, position arena, coin arena
    '''
    arena = game_state['field'] # get np array of arena -1 = wall, 1 = crate
    
    arena_pos = np.zeros_like(arena) # create empty arena just with own postion
    _, _, _, (x, y) = game_state['self'] # get own position
    arena_pos[x,y] = 1

    arena_coins = np.zeros_like(arena) # create empty arena just with coins
    coins = game_state['coins']
    arena_coins[tuple(np.array(coins).T)] = 1

    # delete outside walls ??
    inside = arena[1:-1,1:-1]
    inside_pos = arena_pos[1:-1,1:-1]
    inside_coins = arena_coins[1:-1,1:-1]

    # return everything as one vector
    return np.concatenate((inside, inside_pos, inside_coins)).reshape(-1)
    '''

    # SECOND IDEA: features: empty inside arena with coin as +2 values and own position as 10
    '''
    arena = game_state['field'] # get np array of arena -1 = wall, 1 = crate

    arena_coins = np.zeros_like(arena) # create empty arena for the coins
    coins = game_state['coins']
    arena_coins[tuple(np.array(coins).T)] = 2 # set the value at coin position as +2

    arena_pos = np.zeros_like(arena) # create empty arena just with own postion
    _, _, _, (x, y) = game_state['self'] # get own position
    arena_pos[x,y] = 10
    
    inside_arena = arena[1:-1,1:-1] + arena_coins[1:-1,1:-1] + arena_pos[1:-1,1:-1] # delete walls
    return inside_arena.reshape(-1)
    '''

    # THIRD IDEA: Just pass location of player & coins to learn how to move
    '''
    arena_pos = np.zeros_like(game_state['field']) # create empty arena just with own postion
    _, _, _, (x, y) = game_state['self'] # get own position
    arena_pos[x,y] = 1

    arena_coins = np.zeros_like(arena_pos) # create empty arena just with coins
    coins = game_state['coins']
    arena_coins[tuple(np.array(coins).T)] = 1

    return np.concatenate((arena_pos[1:-1,1:-1], arena_coins[1:-1,1:-1])).reshape(-1)
    '''

    # FOURTH IDEA: Pass location as one-hot and x/y distances to coins
    #               weighted by the total distance in descendin order
    '''
    arena_pos = np.zeros_like(game_state['field']) # create empty arena just with own postion
    _, _, _, (x, y) = game_state['self'] # get own position
    arena_pos[x,y] = 1

    coins = game_state['coins']

    dx = np.zeros(9) # we have 9 coins to collect
    dy = np.zeros(9)
    dist = np.zeros(9) + 1e-10 # to avoid division by 0

    idx = 0
    for coin in coins:
        dx[idx] = coin[0] - x
        dy[idx] = coin[0] - y
        dist[idx] += np.sqrt((coin[0] - x)**2 + (coin[0] - y)**2)
        idx = idx +1

    dx = np.sort(dx/dist, kind='mergesort')[::-1] # reverse order sorted
    dy = np.sort(dy/dist, kind='mergesort')[::-1]

    return np.concatenate((arena_pos[1:-1,1:-1].reshape(-1), dx, dy))
    '''

    # FITH IDEA: similar to fourth ideas but only give dx/dy to nearest coin
    arena_pos = np.zeros_like(game_state['field']) # create empty arena just with own postion
    _, _, _, (x, y) = game_state['self'] # get own position
    arena_pos[x,y] = 1

    coins = game_state['coins']

    dx = np.zeros(9) # we have 9 coins to collect
    dy = np.zeros(9)
    dist = np.zeros(9) + 1e-10 # to avoid division by 0

    idx = 0
    for coin in coins:
        dx[idx] = coin[0] - x
        dy[idx] = coin[0] - y
        dist[idx] += np.sqrt((coin[0] - x)**2 + (coin[0] - y)**2)
        idx = idx +1

    epsilon = 1e-10
    dx = np.sign(dx[np.argmin(np.abs(dx))] + epsilon) * 1/(np.min(np.abs(dx)) + 1) # set number spectrum sign*[3,0.2] 
    dy = np.sign(dy[np.argmin(np.abs(dy))] + epsilon) * 1/(np.min(np.abs(dy)) + 1)

    self.logger.debug(dx)
    self.logger.debug(dy)


    return np.concatenate((arena_pos[1:-1,1:-1].reshape(-1), (dx, dy)))

    ###   ###