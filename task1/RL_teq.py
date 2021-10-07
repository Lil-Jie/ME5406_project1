import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RL(object):
    def __init__(self, actions, learning_rate=0.01, discount_rate=0.9, e_greedy=0.1):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # define the policy that how robot choose its actions
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        self.optimal = 1 - self.epsilon + self.epsilon / 4

        if np.random.uniform() < self.optimal:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # check whether the state is already exist
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def plot_figure(self, epi_num, steps, s_r_c, name):

        # plot the steps over episodes
        plt.figure(2)
        ax1 = plt.subplot(121)
        ax1.plot(np.arange(epi_num), steps, 'blue', linewidth=1)
        plt.title('Steps over episodes-({})'.format(name))
        plt.xlabel('Episodes')
        plt.ylabel('Steps')

        # plot the success rate over episodes
        ax2 = plt.subplot(122)
        ax2.plot(np.arange(epi_num), s_r_c, 'red', linewidth=1)
        plt.title('Success rate over episodes-({})'.format(name))
        plt.xlabel('Episodes')
        plt.ylabel('Success rate (%)')

        plt.show()

# Q-learning with an ϵ-greedy behavior policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.1, discount_rate=0.9, e_greedy=0.1):
        super(QLearningTable, self).__init__(actions, learning_rate, discount_rate, e_greedy)

    def q_update(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'frisbee' or s_ != 'hole':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r                                               # next state is terminal(frisbee or hole)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)     # update the Q value


# SARSA with an ϵ-greedy behavior policy
class SARSATable(RL):

    def __init__(self, actions, learning_rate=0.1, discount_rate=0.9, e_greedy=0.1):
        super(SARSATable, self).__init__(actions, learning_rate, discount_rate, e_greedy)

    def q_update(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'frisbee' or s_ != 'hole':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]    # next state is not terminal
        else:
            q_target = r                                            # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update the Q value


# First-visit Monte Carlo control without exploring starts
class MCTable(RL):
    def __init__(self, actions, learning_rate=0.1, discount_rate=0.9, e_greedy=0.4):
        super(MCTable, self).__init__(actions, learning_rate, discount_rate, e_greedy)

    # calculate discounted return G
    def get_sag(self, sar_list):
        G = 0
        sag_list = []
        for s, a, r in reversed(sar_list):
            G = r + self.gamma * G
            sag_list.append([s, a, G])
        return sag_list
