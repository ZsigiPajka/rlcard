''' An example of evluating the trained models in RLCard
'''
import os
import argparse

import rlcard
from agents.vanilla_dqn.vanilla_dqn import Vanilla_DQN_agent

from rlcard.utils import (
    get_device,
    set_seed,
    tournament, plot_history,
)

def load_model(model_path, env=None, position=None, device=None):
    if model_path == 'experiments/double_dueling_dqn_model':
        agent = Vanilla_DQN_agent()
        agent.load(os.path.join(args.models[1],'ledh_against_random'))
    elif model_path == 'experiments/vanilla_dqn_model':
        agent = Vanilla_DQN_agent()
        agent.load(os.path.join(args.models[1],'ledh_against_random'))
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

def evaluate(args):

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

    # Evaluate
    rewards = tournament(env, args.num_games)
    log_dir='experiments/evaluations/'
    csv_path = os.path.join(log_dir, 'history.csv')
    fig_path = os.path.join(log_dir, 'fig.png')
    # plot_history(csv_path, fig_path, args.names)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)

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
            'experiments/leduc_holdem_cfr_result/cfr_model',
        ],
    )
    parser.add_argument(
        '--names',
        nargs='*',
        default=[
            'external_cfr',
            'double_dueling_dqn',
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

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

