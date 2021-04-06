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
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
gamma = 0.8
lr = 0.1

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
REDUCE_OVERALL_DIST = "REDUCE_OVERALL_DIST"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.counter = 0
    self.logger.debug("Setting up training:")



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

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

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    if (old_game_state != None) and (new_game_state != None):
        dist_old = feature_old[-1]
        dist_new = feature_new[-1]
        if dist_new < dist_old:
            events.append(REDUCE_OVERALL_DIST)

    # state_to_features is defined in callbacks.py
    X_old = state_to_features(self, old_game_state)
    X_new = state_to_features(self, new_game_state)
    reward = reward_from_events(self, events)

    self.transitions.append(Transition(X_old, self_action, X_new, reward))

    #self.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events)))

    # Updating the model: (TD Q-learning)
    if self.counter > 2:
        self.logger.debug('Updating the model:')
        action_max = self.ACTIONS[np.argmax(Q_func(self, new_game_state))]
        Q_max = Q_func(self, new_game_state, action_max)

        target = reward + gamma * Q_max
        index = np.where(self.ACTIONS == self_action)
        index = index[0][0]

        beta_a = self.beta_model[:, index]
        self.logger.debug(f'beta_a before:{beta_a}')
        self.beta_model[:, index] = self.beta_model[:, index] * lr * beta_a * X_old * (target - X_old @ beta_a)
        self.logger.debug(f'beta_a after:{self.beta_model[:, index]}')
        self.logger.debug(f'beta_a difference:{beta_a - self.beta_model[:, index]}')



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
    with open("beta_model.pt", "wb") as file:
        pickle.dump(self.beta_model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        REDUCE_OVERALL_DIST: 30,
        e.MOVED_DOWN: 2,
        e.MOVED_UP: 2,
        e.MOVED_RIGHT: 2,
        e.MOVED_LEFT: 2,
        e.WAITED: -6,
        e.INVALID_ACTION: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
