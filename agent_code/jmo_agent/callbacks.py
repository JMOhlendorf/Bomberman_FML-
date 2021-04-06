import os
import pickle
import numpy as np
from scipy.special import softmax


# number of tiles the agent scans the field around itself, resulting in a (2*scan_length + 1)**2 scanned square
scan_length = 3
# number of feature values
feature_length = ((2 * scan_length) + 1)**2 + 289 + 7

# name of the model to build or load
model_name = "model_new.pt"
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
    that are independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])
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
    self.logger.info("ENTERED State to features")
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
        # 1) feature: surrounding of the agent
        # if scan_length > 1 a new field is created with -1 values around the original field
        #self.logger.info("feat00")
        field = game_state['field']
        #self.logger.info("feat01")
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
        #self.logger.info("feat1")

        # 2) feature: one hot encoded bomb field, countdown times, distance to bombs, x and y coordinates,
        #             bomb boolean
        bombs = game_state['bombs']
        bombs_field = np.zeros_like(field)
        # self.logger.info(f"bombs_field.shape:{bombs_field.shape}")
        # needs fixing if more bombs are involved
        bombs_times = 5 * np.ones(1) # set to 5 if no bombs

        x_y_coord = np.zeros(2) # set to False=0 if no bombs, first value for x

        dist_bombs = 6 * np.ones(1) # set to 6 if no bombs
        #self.logger.info("feat21")
        agent_pos = np.array(game_state['self'][3])
        #self.logger.info(f"type(agent_pos):{type(agent_pos)}")
        #self.logger.info(agent_pos)


        agent_x = agent_pos[0]
        agent_y = agent_pos[1]
        #self.logger.info(f"agent_pos:{agent_x}")
        #self.logger.info("feat22")


        if len(bombs) > 0:
            bombs_pos = np.array(bombs[0][0])
            #self.logger.info(f"type(bombs_pos):{type(bombs_pos)}")
            #self.logger.info(bombs_pos)
            bombs_x = bombs_pos[0]
            bombs_y = bombs_pos[1]

            # one hot encoded bomb field
            bombs_field[bombs_x, bombs_y] = 2
            # countdown times of bomb
            bombs_times[0] = bombs[0][1]

            #self.logger.info(f"type(bombs_pos - agent_pos):{type(np.array(bombs_pos - agent_pos))}")
            #self.logger.info(np.array(bombs_pos - agent_pos))
            dist_bombs_temp = np.linalg.norm(bombs_pos - agent_pos)
            dist_bombs[0] = dist_bombs_temp

            if agent_x == bombs_x:
                x_y_coord[0] = 1
            if agent_y == bombs_y:
                x_y_coord[1] = 1

        bombs_field = bombs_field.flatten()
        #self.logger.info("feat23")

        bomb_bool_temp = game_state['self'][2]
        if bomb_bool_temp == True:
            bomb_bool = np.array([1])
        else:
            bomb_bool = np.array([0])

        # 3) feature: Crate density
        #self.logger.info("feat31")
        crate_mask = (field == 1)
        crate_density = np.array([(np.sum(crate_mask)) / 225])

        # 4) distance to closest coin
        #self.logger.info("feat41")
        closest_coin_dist = np.zeros(1)
        #self.logger.info("feat42")
        coins_pos = np.array(game_state['coins'])
        #self.logger.info("feat43")
        #self.logger.info(f"len(coins_pos):{len(coins_pos)}")
        if len(coins_pos) > 0:
            #self.logger.info("feat51")
            coin_dist = np.linalg.norm(coins_pos - agent_pos, axis=1) + 1
            closest_coin_dist[0] = np.min(coin_dist) / 20
    """
    self.logger.info("feat51")
    self.logger.info(f"surrouding:{surrounding.shape}")
    self.logger.info(f"bombs_field.shape:{bombs_field.shape}")
    self.logger.info(f"bombs_times.shape:{bombs_times.shape}")
    self.logger.info(f"x_y_coord.shape:{x_y_coord.shape}")
    self.logger.info(f"bomb_bool.shape:{bomb_bool.shape}")
    self.logger.info(f"crate_density.shape:{crate_density.shape}")
    self.logger.info(f"closest_coin_dist.shape:{closest_coin_dist.shape}")
    """
    features = np.concatenate((surrounding, bombs_field, bombs_times, x_y_coord, dist_bombs, bomb_bool, crate_density, closest_coin_dist))
    #self.logger.info("feat61")
    #self.logger.info(f"features:{features}")

    return features
