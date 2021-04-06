import os
import pickle
import numpy as np
from scipy.special import softmax


feature_length = 10


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
    self.ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'])

    if self.train or not os.path.isfile("beta_model.pt"):
        self.logger.info("Setting up model from scratch.")
        # possibly normalize
        self.beta_model = np.random.rand(feature_length, len(self.ACTIONS))
        for i in range(len(self.ACTIONS)):
            self.beta_model = self.beta_model / np.sum(self.beta_model[:, i])

    else:
        self.logger.info("Loading model from saved state.")
        with open("beta_model.pt", "rb") as file:
            self.logger.info(f"Opening file: {file}")
            self.beta_model = pickle.load(file)



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    if self.train:
        self.logger.debug("Agent is training:")
        softmax_prob = softmax(Q_func(self, game_state))
        action = np.random.choice(self.ACTIONS, p=softmax_prob)

        return action

    if not self.train:
        self.logger.debug("Trained agent is playing")
        Q = Q_func(self, game_state)
        action = self.ACTIONS[np.argmax(Q)]
        return action


def Q_func(self, game_state, action=None):
    feat_vec = state_to_features(self, game_state)

    if action is not None:
        index = np.where(self.ACTIONS == action)
        index = index[0][0]
        Q = feat_vec @ self.beta_model[:, index]
        return Q

    Q = np.zeros(len(self.ACTIONS))
    for i in range(len(self.ACTIONS)):
        Q[i] = feat_vec @ self.beta_model[:, i]
    return Q


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

    else:
        agent_pos = game_state['self'][3]
        coins_pos = np.array(game_state['coins'])
        dist_coins = np.ones((9))
        dist_coins_temp = np.linalg.norm(coins_pos - agent_pos, axis=1)
        dist_coins[0 : len(dist_coins_temp)] = dist_coins_temp + 1
        total_dist = np.sum(dist_coins) / 190

        field = game_state['field']
        x = game_state['self'][3][0]
        y = game_state['self'][3][1]
        surrounding = []
        for j in range(y - 1, y + 2):
            for i in range(x - 1, x + 2):
                surrounding.append(field[i, j])

    # For example, you could construct several channels of equal shape, ...
    """
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    self.logger.debug(f'stacked_channels shape:{stacked_channels.shape}')
    # and return them as a vector
    return stacked_channels.reshape(-1)
    """
    # make more elegant
    surrounding.append(total_dist)
    channels = surrounding
    channels = np.array(channels)
    channels = channels.flatten()
    return channels

