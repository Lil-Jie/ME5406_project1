from lake_env_4_4 import *
from RL_teq import *
import time
import math


def iteration(epi_num):
    # parameters to evaluate the performance of the algorithm
    steps = []                # record the steps of each episode
    success_steps = []        # record the steps of each success episode
    num_success = 0           # record the successful times after all episode
    t = 0                     # record the sum of the training time
    success_rate_change = []  # record the success rate change
    now = 0                   # record which episode the training is at

    for episode in range(epi_num):
        # initial observation
        s = env.reset()

        start = time.time()  # get the start time of each episode
        i = 0                # update number of steps for each episode

        while True:
            # fresh environment after each step
            # env.refresh()

            # choose action based on observation
            action = RL.choose_action(str(s))

            # take action and then get the next observation and reward
            s_, reward, done = env.move(action)

            # learn from this transition and update the Q value
            RL.q_update(str(s), action, reward, str(s_))

            # swap observation
            s = s_

            i += 1  # steps plus one after one action(update)

            # when robot reach the frisbee, means one successful time
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
                break

    # end of training
    print('----------This is the end of Q-learning training after', epi_num, 'episode----------')
    print('Sum of running time:', t, 's')
    print('The shortest route to reach the frisbee:', str(min(success_steps)))
    print('The longest route to reach the frisbee', str(max(success_steps)))
    print('The average route to reach the frisbee', str(math.ceil(np.mean(success_steps))))
    success_rate = num_success/epi_num
    print('Success rate: {:.2%}'.format(success_rate))

    # draw the heatmap of route so that can show the policy intuitively
    env.heatmap(RL.q_table, name='Qlearning')

    # draw the figures which show the performance of training
    RL.plot_figure(epi_num, steps, success_rate_change, name='Qlearning')
    env.destroy()


if __name__ == "__main__":
    # call for the environment
    env = Lake()
    # input the action to the Q-learning algorithm
    RL = QLearningTable(actions=list(env.action_space))

    # start training
    iteration(100)     # set the number of episode
    env.mainloop()
    print('----------This is the Q-Table----------')
    print(RL.q_table)
