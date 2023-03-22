#!/usr/bin/env python3

import time
import argparse
import warnings

import gymnasium as gym
import gymnasium.utils.play as play
import numpy as np
import matplotlib.pyplot as plt
import torch

from dqn import RainbowDQN


class Hyperparameters:
    """Simple class for holding the hyperparameters"""
    def __init__(self):
        self.gym_id = 'LunarLander-v2'
        self.seed = 314158
        self.buffer_capacity = 20000  # Experience buffer capacity
        self.init_steps = 10000  # Number of initial steps without learning
        self.batch_size = 128
        self.hidden_dim = 128  # Hidden dimension of Q-value network
        self.learning_rate = 1e-3
        self.discount = 0.99
        self.truncation_punishment = 0
        self.total_timesteps = 100000
        self.target_update_freq = 50  # Update frequency of Q-value target network
        self.evaluate_freq = 1000
        self.snapshot_freq = 20000
        self.evaluate_samples = 5
        self.anneal_steps = 90000
        self.epsilon_limit = 0.01
        self.cuda = True
        env = gym.make(self.gym_id)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')


hyperparamters = Hyperparameters()


def set_all_seeds(env, seed):
    """
    Function for setting the same seed in all objects using seed.

    :seed: seed to set
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_agent(args, agent, record: bool = False):
    """
    Function for training an agent.

    Returns one-dimensional array containing results from evaluations.
    """
    results = np.zeros((args.total_timesteps//args.evaluate_freq))
    env = gym.make(args.gym_id)
    agent.reset()
    state, _ = env.reset()
    set_all_seeds(env, args.seed)
    reward = None
    terminal = None
    for step in range(args.total_timesteps):
        if step == args.init_steps:
            start_time = time.time()
        action = agent.get_action(torch.tensor(state).unsqueeze(0).to(args.device))
        next_state, reward, terminal, truncated, _ = env.step(action)
        if truncated: reward -= args.truncation_punishment
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
                      f'Samples per second: {int((step-args.init_steps)/(time.time()-start_time))}          ',
                      end='')
            if record and (step+1) % args.snapshot_freq == 0:
                filename = f'agent_{args.seed}_{step+1}'
                agent.save_qnet('agents/'+filename)
                play_using_agent(args, agent, True, filename)
        if terminal or truncated:
            state, _ = env.reset()
    if record:
        np.save(f'plots/agent_{args.seed}.npy', results)
        fig = plot_scores(results)
        plt.savefig(f'plots/agent_{args.seed}.png')

    return results


def plot_scores(data: np.array):
    """Helper function for plotting evaluation scores over simulation steps"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data)
    ax.yaxis.grid(True)
    ax.set_ylabel('Score')
    ax.set_xlabel('Steps')
    ax.set_xticklabels([str(x*100) for x in [0, 0, 20, 40, 60, 80, 100]])
    return fig


def play_using_agent(args, agent, record: bool=False, recording_name: str=None):
    """
    Function for showcasing or recording (mutually exclusive) given agent.

    :args: hyperparameters of training
    :agent: agent to simulate
    :record: boolean flag for recording; False value will show simulation on window
    :recording_name: if recording, video will be saved with this prefix
    """
    if record: 
        warnings.simplefilter('ignore', UserWarning)
        env = gym.make(args.gym_id, render_mode='rgb_array')
        recorder = gym.wrappers.RecordVideo(env, 'video', name_prefix=recording_name, 
                                            disable_logger=True)
        state, _ = recorder.reset(seed=args.seed)
    else:
        env = gym.make(args.gym_id, render_mode='human')
        state, _ = env.reset(seed=args.seed)
    total_reward = 0
    while True:
        action = agent.get_action(torch.tensor(state).unsqueeze(0).to(args.device))
        if record:
            next_state, reward, terminal, truncated, _ = recorder.step(action)
        else:
            next_state, reward, terminal, truncated, _ = env.step(action)
        if truncated: reward -= args.truncation_punishment
        total_reward += reward
        state = next_state
        if terminal or truncated: break
    if not record: print('Total reward: ', total_reward)


def play_using_human(args):
    """
    Function for manual control of agent in simulation.

    :args: hyperparameters of training
    """
    env = gym.make(args.gym_id, render_mode='rgb_array')
    key_mappings = {
        'w': 2,
        'a': 1,
        'd': 3,
    }

    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        return [rew,]

    plotter = play.PlayPlot(callback, 150, ["reward"])
    play.play(env, fps=10, keys_to_action=key_mappings, callback=plotter.callback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', help='loads agent from specified file in agents/')
    parser.add_argument('-r', '--record', action='store_true', help='flag for recording evaluations')
    args = parser.parse_args()
    agent = RainbowDQN(hyperparamters)
    if args.load is not None:
        agent.load_qnet('agents/'+args.load)
        play_using_agent(hyperparamters, agent, args.record, args.load)
    else:
        train_agent(hyperparamters, agent, args.record)
