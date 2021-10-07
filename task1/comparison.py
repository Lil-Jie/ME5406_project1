from lake_env_4_4 import *
from RL_teq import *
from Q_learning_run import iteration as iter_Q
from SARSA_run import iteration as iter_S
from MC_run import iteration as iter_MC
import matplotlib.pyplot as plt
import time
import math

# call for the environment
env = Lake()
epi_num = 300

# input the actions and states to call for the main algorithm
RL = QLearningTable(actions=list(env.action_space))
steps_q, success_rate_change_q = iter_Q(epi_num, env, RL)

RL = SARSATable(actions=list(env.action_space))
steps_s, success_rate_change_s = iter_S(epi_num, env, RL)

RL = MCTable(actions=list(env.action_space))
steps_m, success_rate_change_m = iter_MC(epi_num, env, RL)

plt.figure(3)
plt.plot(np.arange(epi_num), steps_q, 'blue', linewidth=1)
plt.plot(np.arange(epi_num), steps_s, 'green', linewidth=1)
plt.plot(np.arange(epi_num), steps_m, 'red', linewidth=1)
plt.title('Steps over episodes (three algorithms)')
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.legend(['Q-learning', 'SARSA', 'Monte Carlo'])

plt.figure(4)
plt.plot(np.arange(epi_num), success_rate_change_q, 'blue', linewidth=1)
plt.plot(np.arange(epi_num), success_rate_change_s, 'green', linewidth=1)
plt.plot(np.arange(epi_num), success_rate_change_m, 'red', linewidth=1)
plt.title('Success rate over episodes (three algorithms)')
plt.xlabel('Episodes')
plt.ylabel('Success rate')
plt.legend(['Q-learning', 'SARSA', 'Monte Carlo'])
plt.show()

env.mainloop()
