
import collections
import pickle

from rlcard.utils.utils import *


class OutcomeCFRAgent:
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
        self.policy = collections.defaultdict(np.array)
        self.average_policy = collections.defaultdict(np.array)
        self.policy_sum = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)
        # epsilon for exploration rate
        self.epsilon = 0.14
        self.iteration = 0

    def train(self):
        ''' Do one iteration of CFR
        '''
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.num_players):
            self.env.reset()
            probs = np.ones(self.env.num_players)
            self.traverse_tree(player_id, probs, 1.0)

    def traverse_tree(self, player_id, probs, s):
        ''' Traverse the game tree, update the regrets

        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value
            s: sample probability

        Returns:
            state_utilities (list): The expected utilities for all the players
            ptail: probability of reaching the terminal game node
        '''
        # on terminal node, return payoffs
        if self.env.is_over():
            if s == 0: s = 0.05
            return self.env.get_payoffs() / s, 1.0

        # get current player
        current_player = self.env.get_player_id()
        # get legal actions and state
        obs, legal_actions = self.get_state(current_player)
        # insert state into regrets and policy if not present
        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)
        # get action probability distribution trough regret matching
        action_probs = self.regret_matching(obs)
        # create epsilon policy for this state
        epsilon_policy = collections.defaultdict(np.array)
        epsilon_policy[obs] = np.zeros(self.env.num_actions)
        # traversing player plays according to epsilon policy
        if current_player == player_id:
            for action in range(self.env.num_actions):
                epsilon_policy[obs][action] = (self.epsilon / self.env.num_actions) + (1.0 - self.epsilon) * \
                                              action_probs[action]
            policy = epsilon_policy
        else:
            # for otherplayer uses unchanged policy form dict
            policy = self.policy
        # get action according to policy
        action = self.get_action(obs, legal_actions, policy)
        # play action
        self.env.step(action)
        new_probs = probs.copy()
        # update action probabilities
        new_probs[current_player] *= action_probs[action]
        # get state utility and ptail recursively from all games up to terminal node
        # pass in updated action probabilities and updated sample probability
        state_utility, ptail = self.traverse_tree(player_id, new_probs, s * policy[obs][action])
        # get counterfactual probability
        counterfactual_prob = (np.prod(probs[:current_player]) *
                               np.prod(probs[current_player + 1:]))
        # for traversing player calculate regrets
        if current_player == player_id:
            # get wight
            w = state_utility[player_id] * counterfactual_prob
            for a in legal_actions:
                # for every action calculate its regret and save it into dictionary
                regret = w * (1.0 - action_probs[action]) * ptail if a == action else -w * ptail * action_probs[action]
                self.regrets[obs][action] += regret
        # for other player calculate policy sum
        else:
            if obs not in self.policy_sum:
                self.policy_sum[obs] = np.zeros(self.env.num_actions)
            for action in legal_actions:
                self.policy_sum[obs][action] += (counterfactual_prob / s) * action_probs[action] * self.iteration
        # return state utility and updated ptail to previous state
        return state_utility, (ptail * action_probs[action])

    def update_avg_policy(self):
        '''
        Calculates avg policy, which model uses to choose actions during gameplay
        '''
        # iterate over all states
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

    #
    def get_action(self, obs, legal_actions, policy):
        '''
        Returns action according to policy using wighted random choice
        Args:
            obs (string): The state_str
            legal_actions (list): Indices of legal actions
            policy (dict): The used policy
        Returns:
            action (int): index of selected action
        '''
        probs = self.action_probs(obs, legal_actions, policy)
        action = np.random.choice(len(probs), p=probs)
        return action

    def regret_matching(self, obs):
        ''' Preforms regret matching method on provided state policy

        Args:
            obs (string): The state_str
        '''
        self.policy[obs] = self.regrets[obs]
        for i in range(len(self.regrets[obs])):
            if self.regrets[obs][i] < 0:
                self.policy[obs][i] = 0
        normalizing_sum = sum(self.policy[obs])
        if normalizing_sum > 0:
            self.policy[obs] = self.policy[obs] / normalizing_sum
        else:
            self.policy[obs] = np.repeat(1 / self.env.num_actions, self.env.num_actions)
        return self.policy[obs]

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
