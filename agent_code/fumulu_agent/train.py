import pickle
import os
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 6  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

### ADDED ###

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
LEARNING_RATE = 0.06
DISCOUNT_FACTOR = 0.4

###   ###

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
REPETITION = "REPETITION" # makes no sense
DECREASED_COIN_DISTANCE = "DECREASED_COIN_DISTANCE"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    ### ADDED ###
    feature_length = 15*15 + 2 # has to be adapted when "state_to_features" is altered

    # Initialize betas if non existant yet
    if not os.path.isfile("beta.pt"):
        # Initialize beta vectors for each action (for linear approximation)
        
        beta_up = np.zeros(feature_length)
        beta_right = np.zeros(feature_length)
        beta_down = np.zeros(feature_length)
        beta_left = np.zeros(feature_length)
        beta_wait = np.zeros(feature_length)
        self.beta = np.array([beta_up, beta_right, beta_down, beta_left, beta_wait])
    
    else: 

        ### LOAD BETAs
        print("Load model")
        with open("beta.pt", "rb") as file:
            self.beta = pickle.load(file)

    # Setup global distance to nearest coin variable for rewards
    self.coin_distance = 0

    ###   ###


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)


    ### ADDED ###
    # state_to_features is defined in callbacks.py
    old_state_array = state_to_features(old_game_state)
    new_state_array = state_to_features(new_game_state)

    # Setup reward for step which decreases the NEAREST coin distance
    if old_game_state is not None:
        coins = old_game_state['coins']
        _, _, _, (x_old, y_old) = old_game_state['self']
        _, _, _, (x_new, y_new) = new_game_state['self']
        old_dist, new_dist = 100, 100
        # Get distances to nearest coin
        for coin in coins:
            old_dist_candidate = np.sqrt((x_old - coin[0])**2 + (y_old - coin[1])**2)
            new_dist_candidate = np.sqrt((x_new - coin[0])**2 + (y_new - coin[1])**2)
            if old_dist_candidate < old_dist:
                old_dist = old_dist_candidate
            if new_dist_candidate < new_dist:
                new_dist = new_dist_candidate

        if old_dist > new_dist:
            events.append(DECREASED_COIN_DISTANCE)
            self.coin_distance = new_dist

    # Already existing in given code:
    self.transitions.append(Transition(old_state_array, self_action, new_state_array, reward_from_events(self, events)))



    # Calculate response y using n-step temporal difference with Q learning (or other methods)
    n = TRANSITION_HISTORY_SIZE
    if len(self.transitions) == n and self.transitions[-n][0] is not None:

        # Calculate Q function with linear approximation
        Q = Q_func(self, old_state_array, self_action)

        # Repetition makes no sense since we should only look at presend state to determine best action
        ''' 
        # Add Event if agent moves back and forth
        older_state_array = self.transitions[-2][0]
        if older_state_array is not None and new_state_array is not None and older_state_array.all() == new_state_array.all():
            events.append(REPETITION)
        '''

        # REWARD: TD Q-Learning
        '''
         # --- #
        y = self.transitions[-1][3] + DISCOUNT_FACTOR * np.max(Q_func(self, new_state_array, ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']))
        print("response y = ", y)
        # --- #
        '''

        # REWARD: n-step TD Q-Learning ---> reward for TRANSITION_HISTORY_SIZE steps back <---
        
        # --- #
        y = 0
        for t in range(n):
            y += DISCOUNT_FACTOR**t * self.transitions[t-n][3] # discount factor times rewards from future events
            #print(y)
        print("initial: ", y)
        y += DISCOUNT_FACTOR**n * np.max(Q_func(self, old_state_array, ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']))
        
        # Account for double coin and prevent exploding values
        #if y > 50:
        #    y = 0 
        print("response y = ", y)
        # --- #
        

        '''
        # REWARD: SARSA
        # --- #
        y = self.transitions[-1][3] + DISCOUNT_FACTOR*np.max(Q_func(self, new_state_array, ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']))
        #print("response y = ", y)
        # --- #
        '''


        # update betas (THIS STEP)
        '''
        if self_action == 'UP':
            self.beta[0] = self.beta[0] + LEARNING_RATE * old_state_array * (y - Q)
        if self_action == 'RIGHT':
            self.beta[1] = self.beta[1] + LEARNING_RATE * old_state_array * (y - Q)
        if self_action == 'DOWN':
            self.beta[2] = self.beta[2] + LEARNING_RATE * old_state_array * (y - Q)
            #print("updated beta DOWn: ", self.beta[2])
        if self_action == 'LEFT':
            self.beta[3] = self.beta[3] + LEARNING_RATE * old_state_array * (y - Q)
        if self_action == 'WAIT':
            self.beta[4] = self.beta[4] + LEARNING_RATE * old_state_array * (y - Q)
        '''

        # update betas (TRANSITION_HISTORY_SIZE-steps behind)
        if self_action == 'UP':
            self.beta[0] = self.beta[0] + LEARNING_RATE * self.transitions[-n][0] * (y - Q_func(self, self.transitions[-n][0], self.transitions[-n][1]))
        if self_action == 'RIGHT':
            self.beta[1] = self.beta[1] + LEARNING_RATE * self.transitions[-n][0] * (y - Q_func(self, self.transitions[-n][0], self.transitions[-n][1]))
        if self_action == 'DOWN':
            self.beta[2] = self.beta[2] + LEARNING_RATE * self.transitions[-n][0] * (y - Q_func(self, self.transitions[-n][0], self.transitions[-n][1]))
            #print("updated beta DOWn: ", self.beta[2])
        if self_action == 'LEFT':
            self.beta[3] = self.beta[3] + LEARNING_RATE * self.transitions[-n][0] * (y - Q_func(self, self.transitions[-n][0], self.transitions[-n][1]))
        if self_action == 'WAIT':
            self.beta[4] = self.beta[4] + LEARNING_RATE * self.transitions[-n][0] * (y - Q_func(self, self.transitions[-n][0], self.transitions[-n][1]))
    ###   ###

### ADDED ###
def Q_func(self, game_state_array, action):
    '''
    Input: action can either be a string or a list (then Q for all actions in list is returned)
    Returns Q = X*beta_action with X being the game state (linear value approximation)
    '''
    if action == None:
        return 0

    if isinstance(action, list):
        Q_array = np.zeros(len(action))

        for k in range(len(action)):
            Q_array[k] = game_state_array @ self.beta[ACTIONS.index(action[k])]
        return Q_array

    return game_state_array @ self.beta[ACTIONS.index(action)]
###   ###


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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    ### ADDED ###
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) # Reset transitions
    print(self.beta)

    ###   ###

    '''
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    '''
    print("Store model")
    # Store the betas
    with open("beta.pt", "wb") as file:
        pickle.dump(self.beta, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 40, # 1
        #e.KILLED_OPPONENT: 5,
        #PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad


        ### ADDED ###
        DECREASED_COIN_DISTANCE: 4/((self.coin_distance + 1)), # Moved towards nearest coin (distance .. apart)
        e.MOVED_LEFT: -0.1, # Successfully moved one tile to the left.
        e.MOVED_RIGHT: -0.1, # Successfully moved one tile to the right.
        e.MOVED_UP: -0.1, # Successfully moved one tile up.
        e.MOVED_DOWN: -0.1, # Successfully moved one tile down.
        e.WAITED: -1, # Intentionally didn't act at all.
        e.INVALID_ACTION: -1 # Picked a non-existent action or one that couldn't be executed.
        ###  ###
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
