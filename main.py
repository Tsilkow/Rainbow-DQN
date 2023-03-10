#!/usr/bin/env python3

import os
import math
import random
import time

import gymnasium as gym
import gymnasium.utils.play as play
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import *


class parse_args:
    """Simple class for holding the hyperparameters"""
    def __init__(self):
        self.gym_id = 'LunarLander-v2'
        self.seed = 2137
        self.buffer_capacity = 10000 # Experience buffer capacity
        self.init_steps = 10000 # Number of initial steps without learning
        self.batch_size = 128
        self.hidden_dim = 128 # Hidden dimension of Q-value network
        self.learning_rate = 7e-4
        self.discount = 0.99
        self.total_timesteps = 40000
        self.target_update_freq = 50 # Update frequency of Q-value target network
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
    """Function for setting the same seed in all objects using seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_agent(args, agent):
    """
    Function for training an agent.
    """
    results = np.zeros((args.total_timesteps//args.evaluate_freq))
    env = gym.make(args.gym_id)
    agent.reset()
    state, _ = env.reset(seed=args.seed)
    set_all_seeds(env, args.seed)
    reward = None
    terminal = None
    for step in range(args.total_timesteps):
        if step == args.init_steps:
            start_time = time.time()
        action = agent.get_action(torch.tensor(state).unsqueeze(0).to(args.device))
        next_state, reward, terminal, _, _ = env.step(action)
        agent.buffer.add(state, action, reward, next_state, terminal)
        agent.anneal(step)
        state = next_state
        if step >= args.init_steps:
            agent.update()
            if (step+1) % args.target_update_freq == 0:
                agent.update_target()
            if (step+1) % args.evaluate_freq == 0:
                eval_reward = agent.evaluate(args.evaluate_samples)
                results[step//args.evaluate_freq] = eval_reward
                print(f'\rStep: {step} '
                        f'Evaluation reward: {eval_reward:.2f} '
                        f'Samples per second: {int((step-args.init_steps)/(time.time()-start_time))}',
                        end='')
        if terminal:
            state, _ = env.reset()
            episode_reward = 0
    return results


def play_using_agent(args, agent):
    env = gym.make(args.gym_id, render_mode='rgb_array')
    state, _ = env.reset()
    while True:
        action = agent.get_action(torch.tensor(state).unsqueeze(0).to(args.device))
        next_state, reward, terminal, _, _ = env.step(action)
        state = next_state
        if terminal: break


def play_using_human(args):
    env = gym.make(args.gym_id, render_mode='rgb_array')
    key_mappings = {
        'w': 2,
        'a': 1,
        'd': 3,
    }
    play.play(env, keys_to_action=key_mappings)


agent = RainbowDQN(args)
train_agent(args, agent)
play_using_agent(args, agent)
