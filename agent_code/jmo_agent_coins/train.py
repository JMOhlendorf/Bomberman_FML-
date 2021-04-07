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
PLACEHOLDER_EVENT = "PLACEHOLDER"
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
        dist_old = feature_old[-1]
        dist_new = feature_new[-1]
        if dist_new < dist_old:
            events.append(REDUCE_COIN_DIST)
        if dist_new > dist_old:
            events.append(INCREASE_COIN_DIST)

    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward))

    # n-step TD Q-learning
    n_step = TRANSITION_HISTORY_SIZE
    # updating Q function every n_steps
    if (self.counter % n_step) == 0 and (self.counter > n_step):
        update_nstep(self, n_step)

    # SARSA


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
    if len(beta_index[0]) > 0:
        beta_index = beta_index[0][0]
        beta_a = np.copy(self.model[:, beta_index])
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
        REDUCE_COIN_DIST: 10,
        INCREASE_COIN_DIST: -10,
        e.MOVED_DOWN: 1,
        e.MOVED_UP: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_LEFT: 1,
        e.WAITED: -4,
        e.INVALID_ACTION: -6
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
