import csv
import os

import numpy as np

from rlcard.games.base import Card


def set_seed(seed):
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        if 'torch' in installed_packages:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)


def get_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device


def init_standard_deck():
    ''' Initialize a standard deck of 52 cards

    Returns:
        (list): A list of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res


def init_54_deck():
    ''' Initialize a standard deck of 52 cards, BJ and RJ

    Returns:
        (list): Alist of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    res.append(Card('BJ', ''))
    res.append(Card('RJ', ''))
    return res


def rank2int(rank):
    ''' Get the coresponding number of a rank.

    Args:
        rank(str): rank stored in Card object

    Returns:
        (int): the number corresponding to the rank

    Note:
        1. If the input rank is an empty string, the function will return -1.
        2. If the input rank is not valid, the function will return None.
    '''
    if rank == '':
        return -1
    elif rank.isdigit():
        if int(rank) >= 2 and int(rank) <= 10:
            return int(rank)
        else:
            return None
    elif rank == 'A':
        return 14
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    return None


def elegent_form(card):
    ''' Get a elegent form of a card string

    Args:
        card (string): A card string

    Returns:
        elegent_card (string): A nice form of card
    '''
    suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣', 's': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    rank = '10' if card[1] == 'T' else card[1]

    return suits[card[0]] + rank


def print_card(cards):
    ''' Nicely print a card or list of cards

    Args:
        card (string or list): The card(s) to be printed
    '''
    if cards is None:
        cards = [None]
    if isinstance(cards, str):
        cards = [cards]

    lines = [[] for _ in range(9)]

    for card in cards:
        if card is None:
            lines[0].append('┌─────────┐')
            lines[1].append('│░░░░░░░░░│')
            lines[2].append('│░░░░░░░░░│')
            lines[3].append('│░░░░░░░░░│')
            lines[4].append('│░░░░░░░░░│')
            lines[5].append('│░░░░░░░░░│')
            lines[6].append('│░░░░░░░░░│')
            lines[7].append('│░░░░░░░░░│')
            lines[8].append('└─────────┘')
        else:
            if isinstance(card, Card):
                elegent_card = elegent_form(card.suit + card.rank)
            else:
                elegent_card = elegent_form(card)
            suit = elegent_card[0]
            rank = elegent_card[1]
            if len(elegent_card) == 3:
                space = elegent_card[2]
            else:
                space = ' '

            lines[0].append('┌─────────┐')
            lines[1].append('│{}{}     │'.format(rank, space))
            lines[2].append('│         │')
            lines[3].append('│         │')
            lines[4].append('│    {}   │'.format(suit))
            lines[5].append('│         │')
            lines[6].append('│         │')
            lines[7].append('│       {}│'.format(space, rank))
            lines[8].append('└─────────┘')

    for line in lines:
        print('   '.join(line))


def reorganize(trajectories, payoffs):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player]) - 2, 2):
            if i == len(trajectories[player]) - 3:
                reward = payoffs[player]
                done = True
            else:
                reward, done = 0, False
            transition = trajectories[player][i:i + 3].copy()
            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories


def remove_illegal(action_probs, legal_actions):
    ''' Remove illegal actions and normalize the
        probability vector

    Args:
        action_probs (numpy.array): A 1 dimention numpy array.
        legal_actions (list): A list of indices of legal actions.

    Returns:
        probd (numpy.array): A normalized vector without legal actions.
    '''
    probs = np.zeros(action_probs.shape[0])
    probs[legal_actions] = action_probs[legal_actions]
    if np.sum(probs) == 0:
        probs[legal_actions] = 1 / len(legal_actions)
    else:
        probs /= sum(probs)
    return probs


def multy_tournament(env, num):
    ''' Plays tournament between 2 agents set in env
        every 100 tournaments, record mean reward

        Args:
            env: enviroment with agents for playing
            num: number of games


        Returns:
            rewards (array): recorded mean rewards

        '''
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    rewards = []
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        # record rewards from one game
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
        # every 100 iterations, record mean reward
        if counter % 100 == 0:
            rewards.append(payoffs[1] / 100)
            payoffs = [0 for _ in range(env.num_players)]
    return rewards


def plot_stats(rewards, avg_rewards, save_path, labels):
    '''
    Plot graph from provided array into specified location with specified labels
        Args:
            rewards(list): values to plot
            avg_rewards(list) moving average values
            save_path(str): save location
            labels(str array): labels to display in graph
    '''
    import os
    import matplotlib.pyplot as plt
    # prepare graph parapeters
    fig, ax = plt.subplots(figsize=(15, 7))
    xs = np.arange(len(rewards[0]))
    xm = np.arange(len(avg_rewards[0]))
    colors = ['lightsteelblue', 'palegreen','lightcoral']
    avg_colors = ['mediumblue', 'green', 'crimson']
    # plot values
    for i in range (len(rewards)):
        ax.plot(xs, rewards[i], label='_nolegend_', color=colors[i])
    for i in range(len(avg_rewards)):
        ax.plot(xm, avg_rewards[i], label=labels[i], color=avg_colors[i])
    # name axis
    ax.set_xlabel('episode', fontsize=21)
    ax.set_ylabel('reward', fontsize=21)
    # add labels
    ax.legend(loc='upper right', fontsize=21, fancybox=True, framealpha=0.4)
    # specify tics
    for t in plt.xticks()[1]:
        t.set_fontsize(21)
    for t in plt.yticks()[1]:
        t.set_fontsize(21)
    ax.grid()
    # save graph
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(save_path)


def tournament(env, num, agents, detailed):
    ''' Evaluate the performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.
        agents(str array): names of agents which are used in evaluation
        detailed(bool): parameter that defines, if evaluation should be detailed

    Returns:
        payoffs: A list of average payoffs for each player
        report(str): A string containing number of wins, draws and loses for each agent
    '''
    # prepare csv for all rewards, if detailed
    if detailed:
        log_dir = 'experiments/evaluations/'
        csv_path = os.path.join(log_dir, 'history.csv')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        csv_file = open(csv_path, 'w')
        writer = csv.DictWriter(csv_file, fieldnames=agents)
        writer.writeheader()
        results = np.zeros(3)
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        # record all losses, draws and wins, if detailed
        if detailed:
            writer.writerow({agents[0]: _payoffs[0], agents[1]: _payoffs[1]})
            if _payoffs[0] > 0:
                results[0] += 1
            elif _payoffs[0] == 0:
                results[1] += 1
            else:
                results[2] += 1
        # record rewards from one game
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    # calculate mean reward through all games
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    # if detailed, close csv and construct report string with wins, losses and draws
    if detailed:
        csv_file.close()
        report = agents[0] + ': WINS:' + str(int(results[0])) + ' TIES: ' + str(int(results[1])) + ' LOSSES: ' + str(
            int(results[2]))
        print(report)
        return payoffs, report
    report = ''
    return payoffs, report


def plot_history(csv_path, fig_path):
    '''
       Plot graph from provided csv path into specified location
           Args:
               csv_path(str): path to csv file
               fig_path(str): save location
       '''
    import matplotlib.pyplot as plt
    # open csv to read from
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        a1 = []
        a2 = []
        header = reader.fieldnames
        # read values from csv
        for row in reader:
            a1.append(float(row[header[0]]) if float(row[header[0]]) >= 0 else 0)
            a2.append(float(row[header[1]]) if float(row[header[1]]) >= 0 else 0)
        barWidth = 0.5
        # prepare graph parameters
        fig, ax = plt.subplots(figsize=(27, 8))
        pos = np.arange(len(a1))
        # plot graph
        ax.bar(pos, a1, color='red', width=barWidth,
               label=header[0])
        ax.bar(pos, a2, color='blue', width=barWidth,
               label=header[1])
        # specify tics
        for t in plt.xticks()[1]:
            t.set_fontsize(24)
        for t in plt.yticks()[1]:
            t.set_fontsize(24)
        ax.set_xlabel('episode', fontsize=26)
        ax.set_ylabel('reward', fontsize=26)
        ax.legend(loc='upper right', fontsize=26)
        ax.grid()
        save_dir = os.path.dirname(fig_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(fig_path)


def plot_curve(csv_path, save_path, a):
    ''' Read data from csv file and plot the results
    '''
    import os
    import csv
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['episode']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        # plot graph
        ax.plot(xs, ys, label=a)
        # name axis
        ax.set_xlabel('episode', fontsize=16)
        ax.set_ylabel('reward', fontsize=16)
        ax.legend(fontsize=16)
        # set tics
        for t in plt.xticks()[1]:
            t.set_fontsize(16)
        for t in plt.yticks()[1]:
            t.set_fontsize(16)
        ax.grid()
        # save graph
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_path)
