import collections

import os
import pickle
import random

from rlcard.utils.utils import *


class ExternalCFRAgent():
    ''' Implement CFR (chance sampling) algorithm
    '''

    def __init__(self, env, model_path='./external_cfr_model'):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = False
        self.env = env
        self.model_path = model_path

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.policy_sum = collections.defaultdict(np.array)
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)
        self.epsilon = 0.3
        self.iteration = 0

    def train(self):
        ''' Do one iteration of CFR
        '''
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.num_players):
            self.env.reset()
            self.traverse_tree(player_id)

    def traverse_tree(self, player_id):
        ''' Traverse the game tree, update the regrets

        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value

        Returns:
            state_utilities (list): The expected utilities for all the players
        '''

        if self.env.is_over():
            return self.env.get_payoffs()

        action_utilities = {}
        state_utility = 0
        # state_utility = np.zeros(self.env.num_players)
        current_player = self.env.get_player_id()
        obs, legal_actions = self.get_state(current_player)
        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)

        action_probs = self.regret_matching(obs)
        # action_probs = self.action_probs(obs, legal_actions, self.policy)
        if not current_player == player_id:
            # if obs not in self.policy_sum:
            #     self.policy_sum[obs] = np.zeros(self.env.num_actions)
            action = self.get_action(obs, legal_actions, self.policy)
            self.env.step(action)
            state_utility = self.traverse_tree(player_id)
            # for action in legal_actions:
            #     action_prob = action_probs[action]
            #     self.policy_sum[obs][action] +=  action_prob
            return state_utility

        for action in legal_actions:
            # action_prob = action_probs[action]
            # Keep traversing the child state
            self.env.step(action)
            action_utilities[action] = self.traverse_tree(player_id)
            self.env.step_back()
            state_utility += action_probs[action] * action_utilities[action]

        if obs not in self.policy_sum:
            self.policy_sum[obs] = np.zeros(self.env.num_actions)
        player_state_utility = state_utility[current_player]
        for action in legal_actions:
            regret = action_utilities[action][current_player] - player_state_utility
            self.regrets[obs][action] += regret
            action_prob = action_probs[action]
            self.policy_sum[obs][action] += self.iteration * action_prob
        return state_utility


    def update_avg_policy(self):
        for obs in self.policy_sum:
            for a in range(self.env.num_actions):
                self.average_policy[obs][a] = 0
            normalizing_sum = 0
            for a in range(self.env.num_actions):
                normalizing_sum += self.policy_sum[obs][a]
            for a in range(self.env.num_actions):
                if normalizing_sum > 0:
                    self.average_policy[obs][a] = self.policy_sum[obs][a] / normalizing_sum
                else:
                    self.average_policy[obs][a] = 1.0 / self.env.num_actions
    def get_action(self, obs, legal_actions, policy):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.env.num_actions - 1)
        else:
            probs = self.action_probs(obs, legal_actions, policy)
            action = np.random.choice(len(probs), p=probs)
        return action

    def regret_matching(self, obs):
        ''' Apply regret matching

        Args:
            obs (string): The state_str
        '''
        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])
        action_probs = np.zeros(self.env.num_actions)

        if positive_regret_sum > 0:
            for action in range(self.env.num_actions):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.num_actions):
                action_probs[action] = 1.0 / self.env.num_actions
        self.policy[obs] = action_probs
        return action_probs

    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        if obs not in policy.keys():
            action_probs = np.array([1.0 / self.env.num_actions for _ in range(self.env.num_actions)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), list(state['legal_actions'].keys())

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''
        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.average_policy)
        action = np.random.choice(len(probs), p=probs)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in
                         range(len(state['legal_actions']))}

        return action, info

    def save(self):
        ''' Save model
        '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'), 'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'), 'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'), 'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'), 'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'), 'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'), 'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'), 'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'), 'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()
