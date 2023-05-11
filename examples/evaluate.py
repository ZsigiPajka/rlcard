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


# loading models for evaluations
def load_model(model_path, env=None, position=None, device=None):
    # outside DQN models
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


# prepare environment and return results from multy evaluation
def multy_evaluate(args):
    env = prepare(args)
    rewards = multy_tournament(env, args.num_games)
    return rewards


# evaluate implemented CFR agents against each other and built-in cfr and NFSP network
def evaluate_cfr():
    # prepare agents path for loading
    agents = {
        'experiments/leduc_holdem_cfr_result/outcome_cfr_model': 'outcome_cfr',
        'experiments/leduc_holdem_cfr_result/external_cfr_model': 'external_cfr'
    }
    vs_agents = {
        'experiments/leduc_holdem_cfr_result/outcome_cfr_model': 'outcome_cfr',
        'experiments/leduc_holdem_cfr_result/external_cfr_model': 'external_cfr',
        'experiments/leduc_holdem_cfr_result/cfr_model': 'chance_cfr',
        'random': 'random_agent',
        'experiments/leduc_holdem_nfsp_result/model.pth': 'NFSP',
    }
    # open file for results writing
    f = open("eval_against_cfr.txt", "a")
    # main loop iterates over CFR agents
    for cfr in agents:
        print('---------------------------------------------------------------')
        # second loop iterates over agents which implemented CFR agents are evaluated on
        for opponent in vs_agents:
            if cfr == opponent:
                continue
            print('\nEvaluating ' + agents[cfr] + ' vs ' + vs_agents[opponent])
            # prepare parser with specific model
            args = parser.parse_args(['--models',
                                      cfr,
                                      opponent,
                                      '--names', agents[cfr], vs_agents[opponent],
                                      ])
            # do tournaments between selected models and export graphs with selected name
            rewards, report = evaluation(args, True, prepare(args), agents[cfr] + '_vs_' + vs_agents[opponent] + '.png')
            # write results into txt file
            f.write(agents[cfr] + ' vs ' + vs_agents[opponent] + ' had mean win of ' + str(rewards[0]) + '\n')
            f.write(report + '\n')
    # close file and write message, when evaluation is finished
    f.close()
    print('Stats saved.')


# prepare evaluation between 2 agents and call method to evaluate
def evaluate(args):
    # prepare environment
    env = prepare(args)

    # Pick a right mode and evaluate
    if args.mode == 'standard_detailed':
        evaluation(args, True, env, 'eval.png')
    elif args.mode == 'standard_simple':
        evaluation(args, False, env, 'eval.png')
    elif args.mode == 'cfr_vs_all_DQN':
        make_stats()
    else:
        evaluate_cfr()


# evaluation between 2 agents with option to create a graph
def evaluation(args, detailed, env, figpath):
    # execute tournament
    rewards, report = tournament(env, args.num_games, args.names, detailed)
    # option to create graph containing progress of tournament
    if detailed:
        # prepare file path and name
        log_dir = 'experiments/evaluations/'
        csv_path = os.path.join(log_dir, 'history.csv')
        fig_path = os.path.join(log_dir, figpath)
        # create graph
        plot_history(csv_path, fig_path)
    for position, reward in enumerate(rewards):
        # print results in console
        print(position, args.models[position], reward)
    return rewards, report


def moving_avg(arr, window_size):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1
    return moving_averages


# evaluate all DQN models vs all CFR models and export graphs
def make_stats():
    global args
    # prepare paths for cfr models
    agents = {
        'leduc_holdem_cfr_result/outcome_cfr_model': 'outcome_cfr',
        'leduc_holdem_cfr_result/external_cfr_model': 'external_cfr'
    }
    # prepare paths for DQN models with trained on variants
    trained_on = ['random', 'cfr', 'self']
    dqn_agents = {
        'vanilla_dqn_model': 'vanilla_dqn',
        'double_dqn_model': 'double_dqn',
        'dueling_dqn_model': 'dueling_dqn',
        'double_dueling_dqn_model': 'double_dueling_dqn'
    }
    # open file for writing results
    f = open("eval_against_dqn.txt", "a")

    rewards = []
    avg_rewards = []
    labels = []
    # main loop iterates over CFR agents
    for cfr in agents:
        # second loop iterates over agents which implemented CFR agents are evaluated on
        for agent in dqn_agents:
            # print results in console
            print('---------------------------------------------------------------')
            print('Evaluating ' + agents[cfr] + ' vs ' + dqn_agents[agent] + '\n')
            # loop iterates over DQN trained on variants
            for mode in trained_on:
                # preparing parser
                args = parser.parse_args(['--models',
                                          'experiments/' + cfr,
                                          'experiments/' + agent,
                                          '--names', agents[cfr], dqn_agents[agent] + '_' + mode,
                                          '--type', mode])
                os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
                print('Testing ' + dqn_agents[agent] + ' trained on ' + mode)
                # get reward from tournament against one DQN variant
                reward = multy_evaluate(args)
                # append to array
                rewards.append(reward)
                # create labels to display in graph
                labels.append(dqn_agents[agent] + '_' + mode)
                # calculate total mean reward over 200 partial mean rewards used as points in graphs
                mean = sum(reward) / 200
                # write results to file
                f.write(agents[cfr] + ' vs ' + dqn_agents[agent] + ' tarined on ' + mode + ' had mean win of ' + str(
                    mean) + '\n')
            # prepare path for graph
            log_dir = 'experiments/evaluations/'
            fig_path = os.path.join(log_dir, agents[cfr] + '_vs_' + dqn_agents[agent] + '.png')
            # create graph
            for i in range(len(rewards)):
                avg_rewards.append(moving_avg(rewards[i], 10))
            # print(avg_rewards[0])
            plot_stats(rewards, avg_rewards, fig_path, labels)
            # reset variables for next evaluation
            labels = []
            rewards = []
            avg_rewards = []
    # close file and write message, when evaluation is finished
    f.close()
    print('Stats saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    # create options for parser
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
    parser.add_argument(
        '--mode',
        type=str,
        default='standard_simple',
        choices=[
            'standard_detailed',
            'standard_simple',
            'cfr_vs_all_DQN',
            'cfr_vs_cfr/nfsp'
        ]
    )
    args = parser.parse_args()
    make_stats()
    # put variables into parser
    # args = parser.parse_args(['--models',
    #                           'experiments/leduc_holdem_cfr_result/outcome_cfr_model',
    #                           'experiments/double_dqn_model',
    #                           '--names', 'outcome', 'double_dqn_model_cfr',
    #                           '--type', 'random',
    #                           '--mode', 'standard_simple'
    #                           ]
    #                          )

    # evaluate(args)
