#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import *
from network import *


class DQN:
    """
    Deep Q-value Network
    """
    def __init__(self, args):
        super(DQN, self).__init__()
        self.args = args 
        self.buffer = ExperienceBuffer(self.args)
        self.epsilon = 1
        self.q_net = QNetwork(self.args).to(self.args.device)
        self.q_target = QNetwork(self.args).to(self.args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate, eps=1e-5)
                
    def get_action(self, state, exploration=True):
        """Method for deciding on next action"""
        with torch.no_grad():
            if np.random.sample() < self.epsilon and exploration:
                return np.random.randint(self.args.action_dim)
            else:
                return torch.argmax(self.q_net(state)).item()

    def anneal(self, step):
        """Method for reducing exploration propensity in the network."""
        if step < self.args.anneal_steps:
            self.epsilon = ((self.args.epsilon_limit - 1)/self.args.anneal_steps) * step + 1

    def update(self):
        """Method for performing a backpropagation through the network."""
        states, actions, rewards, next_states, terminals = self.buffer.sample()
        with torch.no_grad():
            q_ns = torch.max(self.q_target(next_states), dim=1)[0].unsqueeze(1)
        q_targets = rewards + (1-terminals) * self.args.discount * q_ns
        
        self.optimizer.zero_grad()
        q_values = self.q_net(states).gather(1, actions)
        loss = nn.functional.smooth_l1_loss(q_values, q_targets)
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        """
        Method for replacing Q-value target network with the newest Q-value 
        network.
        """
        self.q_target.load_state_dict(self.q_net.state_dict())
        
    def evaluate(self, samples):
        with torch.no_grad():
            env_test = gym.make(self.args.gym_id)
            eval_reward = 0
            for i in range(samples):
                state = env_test.reset()
                episode_reward = 0
                while True:
                    action = self.get_action(torch.tensor(state).unsqueeze(0).to(self.args.device), False)
                    next_state, reward, terminal, _ = env_test.step(action)
                    episode_reward += reward
                    state = next_state
                    if terminal:
                        eval_reward += episode_reward/samples
                        break
        return eval_reward
    
    def reset(self):
        self.buffer = ExperienceBuffer(self.args)
        self.epsilon = 1
        self.q_net = QNetwork(self.args).to(self.args.device)
        self.q_target = QNetwork(self.args).to(self.args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate, eps=1e-5)


class RainbowDQN(DQN):
    """
    Rainbow Deep Q-value Network that combines N-step and Prioritized Buffer 
    as well as Dueling and Noisy Q-value Network.
    """
    def __init__(self, args, nstep=3, std=0.2, alpha=0.2, beta=0.2):
        super(RainbowDQN, self).__init__(args)
        self.buffer = RainbowBuffer(args, nstep, alpha, beta)
        self.alpha = alpha
        self.beta = beta
        self.nstep = nstep 
        self.q_net = RainbowQNetwork(args, std).to(args.device)
        self.q_target = RainbowQNetwork(args, std).to(args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate, eps=1e-5)
        self.std = std
        
    def update(self):
        states, actions, rewards, next_states, terminals, idx, weights = self.buffer.sample()
        with torch.no_grad():
            _, indices_of_best = self.q_net(next_states).max(1, keepdim=True)
            q_ns = torch.gather(self.q_target(next_states), -1, indices_of_best)
        q_targets = rewards + (1-terminals) * self.args.discount**self.nstep * q_ns

        self.optimizer.zero_grad()
        q_values = self.q_net(states).gather(1, actions)
        td_errors = nn.functional.smooth_l1_loss(q_values, q_targets, reduction='none')
        loss = torch.mean(td_errors * weights)
        loss.backward()
        self.optimizer.step()
        priorities = td_errors.detach().squeeze().cpu().tolist()
        self.buffer.update_priorities(idx, priorities)
        
    def anneal(self, step):
        if step < self.args.anneal_steps and step > self.args.init_steps:
            self.buffer.alpha = ((1 - self.alpha)/self.args.anneal_steps)*step + self.alpha
            self.buffer.beta = ((1 - self.beta)/self.args.anneal_steps)*step + self.beta
        else:
            pass

    def get_action(self, state, exploration=True):
        return torch.argmax(self.q_net(state)).item()
    
    def reset(self):
        self.buffer = RainbowBuffer(self.args, self.nstep, self.alpha, self.beta) 
        self.q_net = RainbowQNetwork(self.args, self.std).to(self.args.device)
        self.q_target = RainbowQNetwork(self.args, self.std).to(self.args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate, eps=1e-5)