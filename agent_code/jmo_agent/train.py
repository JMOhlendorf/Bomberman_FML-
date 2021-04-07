import pickle
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features, Q_func

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 6  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
gamma = 0.9
# learning rate
lr = 0.01

# Events
INCREASE_BOMB_DIST = "INCREASE_BOMB_DIST"
DECREASE_BOMB_DIST = "DECREASE_BOMB_DIST"
AVOID_BOMB = "AVOID_BOMB"
REDUCE_CRATE_DENSITY = "REDUCE_CRATE_DENSITY"
REDUCE_COIN_DIST = "REDUCE_COIN_DIST"
INCREASE_COIN_DIST = "INCREASE_COIN_DIST"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples

    self.logger.debug("Setting up training.")
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # how many times game_events_occured() has been called
    self.counter = 0



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.counter += 1
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    feature_old = state_to_features(self, old_game_state)
    feature_new = state_to_features(self, new_game_state)

    if (old_game_state != None) and (new_game_state != None):

        # reward if bomb is tried to be avoided or c
        x_coord_old = feature_old[-5]
        x_coord_new = feature_new[-5]
        y_coord_old = feature_old[-4]
        y_coord_new = feature_new[-4]
        bomb_dist_old = feature_new[-3]
        bomb_dist_new = feature_new[-3]
        if bomb_dist_old <= 4:
            dist_increased = (bomb_dist_new > bomb_dist_old)
            dist_decreased = (bomb_dist_new < bomb_dist_old)
            if (x_coord_old == x_coord_new) and dist_increased:
                events.append(INCREASE_BOMB_DIST)
            elif (x_coord_old == x_coord_new) and dist_decreased:
                events.append(DECREASE_BOMB_DIST)
            elif (y_coord_old == y_coord_new) and dist_increased:
                events.append(INCREASE_BOMB_DIST)
            elif (y_coord_old == y_coord_new) and dist_decreased:
                events.append(DECREASE_BOMB_DIST)
            elif (x_coord_old != x_coord_new) and dist_increased:
                events.append(AVOID_BOMB)
            elif (y_coord_old != y_coord_new) and dist_increased:
                events.append(AVOID_BOMB)

        # reward if crate density is decreased
        crate_dens_old = feature_old[-2]
        crate_dens_new = feature_old[-2]
        if crate_dens_new < crate_dens_old:
            events.append(REDUCE_CRATE_DENSITY)

        # reward if coin distance is decreased, if coins are available
        coin_dist_old = feature_old[-1]
        coin_dist_new = feature_new[-1]
        if (coin_dist_old != 0) and (coin_dist_new != 0):
            if coin_dist_new < coin_dist_old:
                events.append(REDUCE_COIN_DIST)
            if coin_dist_new > coin_dist_old:
                events.append(INCREASE_COIN_DIST)


    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward))

    # n-step TD Q-learning
    n_step = TRANSITION_HISTORY_SIZE
    # updating Q function every n_steps
    if (self.counter % n_step) == 0 and (self.counter > n_step):
        update_nstep(self, n_step)


def update_nstep(self, n_step):
    """
    Updating Q-function
    """
    rewards = 0
    for n in range(n_step):
        rewards += gamma**n * self.transitions[n][-1]

    Q_max = np.max(Q_func(self, self.transitions[-1][-2]))
    target = rewards + gamma**n_step * Q_max

    beta_index = np.where(self.ACTIONS == self.transitions[0][1])
    self.logger.info(f"beta_index:{beta_index}")
    if len(beta_index[0]) > 0:
        self.logger.info("ENTERED Update the model")
        beta_index = beta_index[0][0]
        beta_a = np.copy(self.model[:, beta_index])
        self.logger.debug(f'TYPE(self.transitions[0][0]):{type(self.transitions[0][0])}')

        if type(self.transitions[0][0]) == np.ndarray:
            X_st = self.transitions[0][0]
        else:
            X_st = state_to_features(self, self.transitions[0][0])

        self.model[:, beta_index] = beta_a + (lr * X_st.T * (target - X_st @ beta_a))

        self.logger.debug(f'beta_a difference:{np.sum(np.abs(beta_a - self.model[:, beta_index]))}')


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open(self.model_name, "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        INCREASE_BOMB_DIST: 5,
        DECREASE_BOMB_DIST: -5,
        AVOID_BOMB: 8,
        REDUCE_CRATE_DENSITY: 5,
        REDUCE_COIN_DIST: 5,
        INCREASE_COIN_DIST: -5,
        e.BOMB_DROPPED: 5,
        e.KILLED_SELF: -10,
        e.SURVIVED_ROUND: 10,
        e.MOVED_DOWN: 0.5,
        e.MOVED_UP: 0.5,
        e.MOVED_RIGHT: 0.5,
        e.MOVED_LEFT: 0.5,
        e.WAITED: 0,
        e.INVALID_ACTION: -6
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
