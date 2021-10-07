from lake_env_10_10 import *
from RL_teq import *
from itertools import product
import numpy as np
import time
import math


# Function for generate one episode
def one_episode():
    episode_sar = []                  # store (s,a,r) of each episode
    s = env.reset()                   # initial observation

    start = time.time()               # get the start time of one episode
    steps = []                        # record the steps of one episode
    i = 0                             # update number of steps for one episode
    success_steps = []                # record the steps of this episode if success
    num_success = 0                   # record whether this episode is successful
    t = 0                             # record the time of one episode

    while True:
        # env.refresh()                             # refresh the environment
        action = RL.choose_action(str(s))           # choose the action based on observation
        s_, reward, done = env.move(action)         # take an action and get the next observation and reward
        episode_sar.append([s, action, reward])     # append (St,At+1,Rt+1) into an numpy array
        s = s_                                      # swap the observations

        i += 1  # steps plus one after one action(update)

        # when robot reach the frisbee, means one successful time
        if s == 'frisbee':
            num_success += 1
            success_steps += [i]

        # break while loop when end of this episode
        if done:
            steps += [i]       # steps of each episode
            end = time.time()  # get the end time od each episode
            t += end - start   # the running time for each episode
            break
    return episode_sar, t, num_success, success_steps, steps


def iteration(epi_num):
    # All the return(St,At)
    returns = {(s, a): [] for s, a in product(env.state_space, env.action_space)}

    # parameters to evaluate the performance of the algorithm
    steps_ = []                 # record the steps of each episode
    success_steps_ = []         # record the steps of each success episode
    num_success_ = 0            # record the successful times after all episode
    t_ = 0                      # record the sum of the training time
    success_rate_change = []    # record the success rate change
    now = 0                     # record which episode the training is at

    for episode in range(epi_num):
        # record the exist state and action pairs for each episode
        sa_pairs = []

        # run every episode
        episode_sar, t, num_success, success_steps, steps = one_episode()
        sag_list = RL.get_sag(episode_sar)  # transfer (s,a,r) to (s,a,G)

        t_ += t                                      # summed time
        num_success_ += num_success                  # summed success times
        success_steps_ += success_steps              # append each success steps into a list
        steps_ += steps                              # append steps of each episode into a list
        now += 1
        success_rate_change += [num_success_ / now]  # get the success rate in real time
        print(now, '/', epi_num)                     # show which episode of training is in)

        # Use S,A,G to update Q (first-visit method), returns and sa_paris
        for s, a, G in sag_list:
            sa = (str(s), a)
            # if state and action pairs cannot be found in sa_pairs, means first-visit
            if sa not in sa_pairs:
                returns[sa].append(G)                             # append the current G value to the first-visit (s,a) during each episode, which will be an array
                RL.q_table.loc[str(s), a] = np.mean(returns[sa])  # Q value will be the empirical mean return
                sa_pairs.append(sa)                               # supplement the appeared (s,a)
    # return steps_, success_rate_change

    # end of training
    print('----------This is the end of Monte Carlo training after', epi_num, 'episode----------')
    print('Sum of running time:', t_, 's')
    # print('The shortest route to reach the frisbee:', str(min(success_steps_)))
    # print('The longest route to reach the frisbee', str(max(success_steps_)))
    # print('The average route to reach the frisbee', str(math.ceil(np.mean(success_steps_))))
    success_rate = num_success_ / epi_num
    print('Success rate: {:.2%}'.format(success_rate))

    # draw the heatmap of route so that can show the policy intuitively
    env.heatmap(RL.q_table, name='Monte Carlo')

    # draw the figures which show the performance of training
    RL.plot_figure(epi_num, steps_, success_rate_change, name='Monte Carlo')


# Function of the validation of the quality that how trained Q-Table works
def validation():
    # parameters for validation
    n_s_val = 0  # record the successful times after all validation episode

    # follow the maximum value of Q to choose action
    RL.epsilon = 0.2

    for episode in range(50):
        # initial observation
        s = env.reset()

        while True:
            action = RL.choose_action(str(s))    # choose the action based on observation
            s_, reward, done = env.move(action)  # take an action and get the next observation and reward
            s = s_                               # swap the observations

            # when robot reach the frisbee, means one successful time
            if s == 'frisbee':
                n_s_val += 1

            # break while loop when end of this episode
            if done:
                break

    # end of validation
    print('----------This is the end of Monte Carlo validation for following 50 episode----------')
    s_r_val = n_s_val / 50
    print('Success rate: {:.2%}'.format(s_r_val))
    env.destroy()


if __name__ == "__main__":
    # call for the environment
    env = Lake()
    # input the actions and states to call for the main algorithm
    RL = MCTable(actions=list(env.action_space))

    # start training
    iteration(50000)     # set the number of episode
    env.mainloop()
    print('----------This is the Q-Table----------')
    print(RL.q_table)
    validation()
