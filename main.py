#!/usr/bin/env python3

import os
import math
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import *


class parse_args:
    """Simple class for holding the hyperparameters"""
    def __init__(self):
        self.gym_id = 'LunarLander-v2'
        self.buffer_cap = 10000 # Experience buffer capacity
        self.init_steps = 10000
        self.batch_size = 128
        self.hidden_dim = 128
        self.learning_rate = 7e-4
        self.discount = 0.99
        self.samples = 3
        self.total_timesteps = 40000
        self.target_update_freq = 50
        self.evaluate_freq = 1000
        self.evaluate_samples = 5
        self.anneal_steps = 30000
        self.epsilon_limit = 0.01
        self.cuda = True
        env = gym.make(self.gym_id)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        
args = parse_args()


def set_all_seeds(env, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)



