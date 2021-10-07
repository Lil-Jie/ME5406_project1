from lake_env_10_10 import *
from RL_teq import *
import time
import math


def iteration(epi_num):
    # parameters to evaluate the performance of the algorithm
    steps = []                  # record the steps of each episode
    success_steps = []          # record the steps of each success episode
    num_success = 0             # record the successful times after all episode
    t = 0                       # record the sum of the training time
    success_rate_change = []    # record the success rate change
    now = 0                     # the current time of episode

    for episode in range(epi_num):
        # initial observation
        s = env.reset()
        a = RL.choose_action(str(s))

        start = time.time()             # get the start time of each episode
        i = 0                           # update number of steps for each episode

        while True:
            # fresh environment
            # env.refresh()

            # take action and get next observation and reward
            s_, reward, done = env.move(a)

            # choose action based on next observation
            a_ = RL.choose_action(str(s_))

            # learn from this transition and update Q value
            RL.q_update(str(s), a, reward, str(s_), str(a_))

            # swap observation and action
            s = s_
            a = a_

            i += 1  # steps plus one after one action(update)

            # when robot reach the frisbee, means one successful time and record the steps to reach the frisbee
            if s == 'frisbee':
                num_success += 1
                success_steps += [i]

            # break while loop when end of this episode
            if done:
                steps += [i]                                # steps of each episode
                end = time.time()                           # get the end time od each episode
                t += end - start                            # the running time for each episode
                now += 1
                success_rate_change += [num_success / now * 100]  # get the success rate in real time
                print(now, '/', epi_num)                    # show which episode of training is in
                break
    # return steps, success_rate_change
    
    # end of training
    print('----------This is the end of SARSA training after', epi_num, 'episode----------')
    print('Sum of running time:', t, 's')
    print('The shortest route to reach the frisbee:', str(min(success_steps)))
    print('The longest route to reach the frisbee', str(max(success_steps)))
    print('The average route to reach the frisbee', str(math.ceil(np.mean(success_steps))))
    success_rate = num_success / epi_num
    print('Success rate: {:.2%}'.format(success_rate))

    # draw the heatmap of route so that can show the policy intuitively
    env.heatmap(RL.q_table, name='SARSA')

    # draw the figures which show the performance of training
    RL.plot_figure(epi_num, steps, success_rate_change, name='SARSA')


# Function of the validation of the quality that how trained Q-Table works
def validation():
    # parameters for validation
    n_s_val = 0  # record the successful times after all validation episode

    # follow the maximum value of Q to choose action
    RL.epsilon = 0.1

    for episode in range(50):
        # initial observation
        s = env.reset()
        a = RL.choose_action(str(s))

        while True:
            # take action and get next observation and reward
            s_, reward, done = env.move(a)

            # choose action based on next observation
            a_ = RL.choose_action(str(s_))

            # swap observation and action
            s = s_
            a = a_

            # when robot reach the frisbee, means one successful time
            if s == 'frisbee':
                n_s_val += 1

            # break while loop when end of this episode
            if done:
                break

    # end of validation
    print('----------This is the end of SARSA validation for following 50 episode----------')
    s_r_val = n_s_val / 50
    print('Success rate: {:.2%}'.format(s_r_val))
    env.destroy()


if __name__ == "__main__":
    # call for the environment
    env = Lake()
    # input the action to the SARSA algorithm
    RL = SARSATable(actions=list(env.action_space))

    # start training
    iteration(2000)      # set the number of episode
    print('----------This is the Q-Table----------')
    print(RL.q_table)
    validation()
    env.mainloop()
