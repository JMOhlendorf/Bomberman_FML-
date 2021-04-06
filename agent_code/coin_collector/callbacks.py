import os
import pickle
import random

import numpy as np

import logging
import sys


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

NUM_ACTIONS = len(ACTIONS)

NUM_LOOK_AROUND = 4
NUM_FEATURES = 2*(2*NUM_LOOK_AROUND+1)*(2*NUM_LOOK_AROUND+1) + 4

EPSILON_PLAY = 0.35

def setup(self):
    """
    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    with open("weights.pt", "rb") as file:
        self.weights = pickle.load(file)
            
    
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    if random.random() < 0.35:
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .1]) 
    
    action_map, _ = normalize_state(game_state)
    features = state_to_features(game_state)
    q_values = np.dot(self.weights, features)
    return action_map(ACTIONS[np.argmax(q_values)])
    
def normalize_state(game_state):
    """
    :param game_state: The dictionary that to normalize (in-place).
    
    :return: action_map: function to map action in normalized state to action in input_state,
    reverse_action_map: function to map action in input_state to action in normalized state.
    
    """
   
    if game_state == None:
        return lambda a: a, lambda a: a
    
    agent_x, agent_y = game_state['self'][3]
    
    def flip_tuple_x(t):
        return (16 - t[0], t[1])
        
    def flip_tuple_y(t):
        return (t[0], 16 - t[1])
   
    flipped_x = False
    if agent_x > 8:
        game_state['field'] = np.flipud(game_state['field'])
        game_state['bombs'] = [(flip_tuple_x(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.flipud(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple_x(coin) for coin in game_state['coins']]
        name, score, canPlaceBomb, pos = game_state['self']
        game_state['self'] = (name, score, canPlaceBomb, flip_tuple_x(pos))
        game_state['others'] = [(name, score, canPlaceBomb, flip_tuple_x(pos)) for name, score, canPlaceBomb, pos in game_state['others']]
        flipped_x = True

    flipped_y = False
    if agent_y > 8:
        game_state['field'] = np.fliplr(game_state['field'])
        game_state['bombs'] = [(flip_tuple_y(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.fliplr(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple_y(coin) for coin in game_state['coins']]
        name, score, canPlaceBomb, pos = game_state['self']
        game_state['self'] = (name, score, canPlaceBomb, flip_tuple_y(pos))
        game_state['others'] = [(name, score, canPlaceBomb, flip_tuple_y(pos)) for name, score, canPlaceBomb, pos in game_state['others']]
        flipped_y = True
        
    agent_x_update, agent_y_update = game_state['self'][3]
    
    def transpose_tuple(t):
        return (t[1], t[0])
    
    transposed_board = False
    if agent_y_update > agent_x_update:
        game_state['field'] = np.transpose(game_state['field'])
        game_state['coins'] = [transpose_tuple(coin) for coin in game_state['coins']]
        name, score, canPlaceBomb, pos = game_state['self']
        game_state['self'] = (name, score, canPlaceBomb, transpose_tuple(pos))
        transposed_board = True

    def action_map(a):
        if transposed_board:
            if a == 'RIGHT':
                a = 'DOWN'
            elif a == 'DOWN':
                a = 'RIGHT'
            elif a == 'LEFT':
                a = 'UP'
            elif a == 'UP':
                a = 'LEFT'
        if flipped_x:
            a = 'RIGHT' if a == 'LEFT' else ('LEFT' if a == 'RIGHT' else a)
        if flipped_y:
            a = 'UP' if a == 'DOWN' else ('DOWN' if a == 'UP' else a)
        return a
        
    def reverse_action_map(a):
        if flipped_x:
            a = 'RIGHT' if a == 'LEFT' else ('LEFT' if a == 'RIGHT' else a)
        if flipped_y:
            a = 'UP' if a == 'DOWN' else ('DOWN' if a == 'UP' else a)
        if transposed_board:
            if a == 'RIGHT':
                a = 'DOWN'
            elif a == 'DOWN':
                a = 'RIGHT'
            elif a == 'LEFT':
                a = 'UP'
            elif a == 'UP':
                a = 'LEFT'
        return a
        
    return action_map, reverse_action_map

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector. 
    :param game_state:  A dictionary describing the current game board.
    :return: np.array (NUM_FEATURES,)
    """
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
       
    self_x, self_y = game_state['self'][3]
    
    wall_map = np.zeros((31, 31)) 
    field = game_state['field'] 
    for x in np.arange(17):
        for y in np.arange(17):
            if (field[x, y] != -1):
                continue 
            
            x_rel, y_rel = x - self_x, y - self_y
            wall_map[15 + y_rel, 15 + x_rel] = 1
            
    coin_map = np.zeros((31, 31)) 
    coins = game_state['coins']
    for x, y in coins:
        x_rel, y_rel = x - self_x, y - self_y
        coin_map[15 + y_rel, 15 + x_rel] = 1
        
    coins_in_quartal = [np.sum(coin_map[0:16,0:16]), np.sum(coin_map[0:16,16:32]), np.sum(coin_map[16:32,0:16]), np.sum(coin_map[16:32,16:32])]
    channels = [wall_map[11:20,11:20], coin_map[11:20,11:20]]

    max_coin_quartal = np.zeros(4) 
    max_coin_quartal[np.argmax(coins_in_quartal)] = 1
    
    return np.append(np.stack(channels).reshape(-1), max_coin_quartal)

