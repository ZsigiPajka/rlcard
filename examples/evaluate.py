''' An example of evluating the trained models in RLCard
'''
import collections
import os
import argparse

import rlcard
from agents.vanilla_dqn.vanilla_dqn import Vanilla_DQN_agent

from rlcard.utils import (
    get_device,
    set_seed,
    tournament, plot_history, multy_tournament, plot_stats,
)


def load_model(model_path, env=None, position=None, device=None):
    if model_path == 'experiments/double_dueling_dqn_model':
        agent = Vanilla_DQN_agent()
        agent.load(os.path.join(args.models[1], 'ledh_against_' + args.type))
    elif model_path == 'experiments/vanilla_dqn_model':
        agent = Vanilla_DQN_agent()
        agent.load(os.path.join(args.models[1], 'ledh_against_' + args.type))
    elif model_path == 'experiments/dueling_dqn_model':
        agent = Vanilla_DQN_agent()
        agent.load(os.path.join(args.models[1], 'ledh_against_' + args.type))
    elif model_path == 'experiments/double_dqn_model':
        agent = Vanilla_DQN_agent()
        agent.load(os.path.join(args.models[1], 'ledh_against_' + args.type))
    elif os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]

    return agent


def prepare(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)
    return env


def multy_evaluate(args):
    env = prepare(args)
    # print(args.models, args.names. args.type)
    rewards = multy_tournament(env, args.num_games)
    return rewards


def evaluate(args, detailed):
    env = prepare(args)

    # Evaluate
    rewards = tournament(env, args.num_games, args.names, detailed)
    if detailed:
        log_dir = 'experiments/evaluations/'
        csv_path = os.path.join(log_dir, 'history.csv')
        fig_path = os.path.join(log_dir, 'eval.png')
        plot_history(csv_path, fig_path)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)


def make_stats():
    global args
    agents = {
        'leduc_holdem_cfr_result/outcome_cfr_model': 'outcome_cfr',
        'leduc_holdem_cfr_result/external_cfr_model': 'external_cfr'
    }
    trained_on = ['random', 'cfr', 'self']
    dqn_agents = {
        'vanilla_dqn_model': 'vanilla_dqn',
        'double_dqn_model': 'double_dqn',
        'dueling_dqn_model': 'dueling_dqn',
        'double_dueling_dqn_model': 'double_dueling_dqn'
    }
    rewards = []
    labels = []
    for cfr in agents:
        for agent in dqn_agents:
            print('---------------------------------------------------------------')
            print('Evaluating ' + agents[cfr] + ' vs ' + dqn_agents[agent] + '\n')
            for mode in trained_on:
                args = parser.parse_args(['--models',
                                          'experiments/' + cfr,
                                          'experiments/' + agent,
                                          '--names', agents[cfr], dqn_agents[agent] + '_' + mode,
                                          '--type', mode])
                os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
                print('Testing ' + dqn_agents[agent] + ' trained on ' + mode)
                rewards.append(multy_evaluate(args))
                labels.append(dqn_agents[agent] + '_' + mode)
            log_dir = 'experiments/evaluations/'
            fig_path = os.path.join(log_dir, agents[cfr] + '_vs_' + dqn_agents[agent] + '.png')
            plot_stats(rewards, fig_path, labels)
            labels = []
            rewards = []
    print('Stats saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            'experiments/leduc_holdem_cfr_result/external_cfr_model',
            'random',
        ],
    )
    parser.add_argument(
        '--names',
        nargs='*',
        default=[
            'outcome',
            'random',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=20000,
    )

    parser.add_argument(
        '--type',
        type=str,
        default=None,
    )

    # make_stats()
    args = parser.parse_args(['--models',
                              'experiments/leduc_holdem_cfr_result/external_cfr_model',
                              'random',
                              '--names', 'outcome', 'random']
                             )
    evaluate(args, True)