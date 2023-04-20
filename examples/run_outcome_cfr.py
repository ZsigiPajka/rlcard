''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

import rlcard
from agents.vanilla_dqn.vanilla_dqn import Vanilla_DQN_agent
from rlcard.agents import (
    OutcomeCFRAgent,
    RandomAgent,
)
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)


def train(args):
    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initilize CFR Agent
    agent = OutcomeCFRAgent(
        env,
        os.path.join(
            args.log_dir,
            'outcome_cfr_model',
        ),
    )
    agent.load()  # If we have saved model, we first load the model
    agent2 = Vanilla_DQN_agent()
    agent2.load(os.path.join('experiments/double_dqn_model/ledh_against_random'))
    # Evaluate CFR against random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            agent.train()
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                agent.update_avg_policy()
                agent.save()  # Save model
                rewards, report = tournament(
                    eval_env,
                    args.num_eval_games,
                    ['outcome', 'random'],
                    False
                )
                logger.log_performance(
                    episode,
                    rewards[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'outcome_cfr')


if __name__ == '__main__':
    # create options for parser
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=14000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_cfr_result/',
    )

    args = parser.parse_args()
    # train model with options from parser
    train(args)
