import os
import pickle
import numpy as np
from scipy.special import softmax


# number of tiles the agent scans the field around itself, resulting in a (2*scan_length + 1)**2 scanned square
scan_length = 3
# number of feature values
#feature_length = ((2 * scan_length) + 1)**2 + 289 + 2
feature_length = ((2 * scan_length) + 1)**2 + 2
# name of the model to build or load
model_name = "model_rounds500.pt"
# temperature parameter for softmax
rho = 2

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
    self.model_name = model_name

    if self.train or not os.path.isfile(model_name):
        self.logger.info("Setting up model from scratch.")
        # initialize model weights
        self.model = np.random.rand(feature_length, len(self.ACTIONS))
        # normalize model weights
        for i in range(len(self.ACTIONS)):
            self.model = self.model / np.sum(self.model[:, i])

    else:
        self.logger.info("Loading saved model.")
        with open(model_name, "rb") as file:
            self.logger.info(f"Type of the file: {type(file)}")
            self.model = pickle.load(file)



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # i) softmax policy using temperature
    if self.train or not os.path.isfile(model_name):
        # self.logger.debug("Agent is training:")
        softmax_prob = softmax(Q_func(self, game_state) / rho)
        action = np.random.choice(self.ACTIONS, p=softmax_prob)
        return action

    # using trained model to determine actions
    if not self.train:
        # self.logger.debug("Trained agent is playing")
        """
        # alternative action calculation with randomness
        Q = Q_func(self, game_state)
        Q_prob = softmax(Q)
        action = np.random.choice(self.ACTIONS, p=Q_prob)
        return action
        """

        Q = Q_func(self, game_state)
        self.logger.info(f"Q: {Q}")
        action = self.ACTIONS[np.argmax(Q)]
        self.logger.info(f"Took the action: {action}")
        return action



def Q_func(self, game_state):
    """
    Calculates the Q function for the given state
    """

    feat_vec = state_to_features(self, game_state)

    Q = np.zeros(len(self.ACTIONS))
    for i in range(len(self.ACTIONS)):
        Q[i] = feat_vec @ self.model[:, i]

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
        return

    else:
        # feature: surrounding of the agent
        # if scan_length > 1 a new field is created with -1 values around the original field
        field = game_state['field']
        #self.logger.info(f"FIELD: {field}")
        added_tiles = scan_length - 1
        new_shape = 17 + 2 * added_tiles
        new_field = -1 * np.ones((new_shape, new_shape))
        # filling the new_field with original tile values
        new_field[added_tiles: 17 + added_tiles, added_tiles: 17 + added_tiles] = field

        x = game_state['self'][3][0] + added_tiles
        y = game_state['self'][3][1] + added_tiles
        surrounding = []
        for j in range(y - scan_length, y + (scan_length + 1)):
            for i in range(x - scan_length, x + (scan_length + 1)):
                surrounding.append(new_field[i, j])
        surrounding = np.array(surrounding)

        # feature: position/distance of the coins
        agent_pos = game_state['self'][3]
        coins_pos = np.array(game_state['coins'])
        dist_coins = np.zeros((9))
        dist_coins_temp = np.linalg.norm(coins_pos - agent_pos, axis=1)

        dist_coins[0: len(dist_coins_temp)] = dist_coins_temp
        mask = dist_coins > 0.0

        # take only the closest coin as a feature
        closest_dist = np.array([np.min(dist_coins[mask]) / 20])

        # take the total dist as a feature
        # total_dist = np.sum(dist_coins) / 190

        # take steps as a feature
        steps = np.array([game_state['step'] / 20])

        # Using coin positions
        coin_field = np.zeros_like(field)
        coins_x = coins_pos[:, 0]
        coins_y = coins_pos[:, 1]
        coin_field[coins_x, coins_y] = 0.5
        coin_field = coin_field.flatten()

    #features = np.concatenate((surrounding, steps, coin_field, closest_dist))
    features = np.concatenate((surrounding, steps, closest_dist))

    return features
