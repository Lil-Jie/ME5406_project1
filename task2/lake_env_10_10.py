import numpy as np
import time
import tkinter as tk
import random
import seaborn as sns
import matplotlib.pyplot as plt


UNIT = 60   # pixels
Lake_H = 10  # grid height
Lake_W = 10  # grid width


class Lake(tk.Tk, object):
    def __init__(self):
        super(Lake, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.state_space = []
        self.title('Lake_10*10')
        self.geometry('600x600')
        self.build_lake()

    def build_lake(self):
        self.canvas = tk.Canvas(self, bg='white', height=Lake_H * UNIT, width=Lake_W * UNIT)

        # create lake
        self.lake_img = tk.PhotoImage(file='frozen_lake.gif')
        self.lake = self.canvas.create_image(0, 0, anchor='nw', image=self.lake_img)

        # create grids
        for c in range(0, Lake_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, Lake_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, Lake_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, Lake_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # Create origin
        origin = np.array([30, 30])

        # Create robot
        self.robot_img = tk.PhotoImage(file='robot.gif')
        self.robot = self.canvas.create_image(0.5 * UNIT, 0.5 * UNIT, anchor='center', image=self.robot_img)

        # Create frisbee
        self.frisbee_img = tk.PhotoImage(file='frisbee.gif')
        self.frisbee = self.canvas.create_image(9.5 * UNIT, 9.5 * UNIT, anchor='center', image=self.frisbee_img)

        # Create 25 random holes
        random.seed(1999)      # fix random results 4,5,6,8,28
        self.all_holes = []    # store the coordinates and created image of all the created holes
        self.hole_space = []   # store the coordinates where can set holes
        self.state_row = []

        for m in range(10):
            for n in range(10):
                self.state_space.append(str([UNIT * (m + 0.5), UNIT * (n + 0.5)]))   # get coordinates of all the states column by column
                self.state_row.append(str([UNIT * (n + 0.5), UNIT * (m + 0.5)]))     # get coordinates of all the states row by row
                self.hole_space.append([UNIT * (m + 0.5), UNIT * (n + 0.5)])
        # avoid overlap between robot&frisbee and hole
        del self.hole_space[0]
        del self.hole_space[98]

        # sample 25 coordinates randomly from holes_space for holes
        self.holes_co = random.sample(self.hole_space, 25)
        i = 0
        self.hole_img = tk.PhotoImage(file='hole.gif')
        while i < 25:
            self.hole = self.canvas.create_image(self.holes_co[i][0], self.holes_co[i][1], anchor='center', image=self.hole_img)
            self.all_holes.append(self.hole)
            i += 1

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        # time.sleep(0.5)
        self.canvas.delete(self.robot)
        origin = np.array([30, 30])
        self.robot_img = tk.PhotoImage(file='robot.gif')
        self.robot = self.canvas.create_image(0.5 * UNIT, 0.5 * UNIT, anchor='center', image=self.robot_img)
        # return observation
        return self.canvas.coords(self.robot)

    def move(self, action):
        s = self.canvas.coords(self.robot)
        base_action = np.array([0, 0])
        if action == 'up':   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 'down':   # down
            if s[1] < (Lake_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 'right':   # right
            if s[0] < (Lake_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 'left':   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.robot, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.robot)  # next state

        # reward function
        if s_ == self.canvas.coords(self.frisbee):
            reward = 1
            done = True
            s_ = 'frisbee'
        elif s_ in [self.canvas.coords(self.all_holes[0]), self.canvas.coords(self.all_holes[1]), self.canvas.coords(self.all_holes[2]),
                    self.canvas.coords(self.all_holes[3]), self.canvas.coords(self.all_holes[4]), self.canvas.coords(self.all_holes[5]),
                    self.canvas.coords(self.all_holes[6]), self.canvas.coords(self.all_holes[7]), self.canvas.coords(self.all_holes[8]),
                    self.canvas.coords(self.all_holes[9]), self.canvas.coords(self.all_holes[10]), self.canvas.coords(self.all_holes[11]),
                    self.canvas.coords(self.all_holes[12]), self.canvas.coords(self.all_holes[13]), self.canvas.coords(self.all_holes[14]),
                    self.canvas.coords(self.all_holes[15]), self.canvas.coords(self.all_holes[16]), self.canvas.coords(self.all_holes[17]),
                    self.canvas.coords(self.all_holes[18]), self.canvas.coords(self.all_holes[19]), self.canvas.coords(self.all_holes[20]),
                    self.canvas.coords(self.all_holes[21]), self.canvas.coords(self.all_holes[22]), self.canvas.coords(self.all_holes[23]),
                    self.canvas.coords(self.all_holes[24])]:
            reward = -1
            done = True
            s_ = 'hole'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def refresh(self):
        time.sleep(0.01)
        self.update()

    def heatmap(self, Q, name):
        h_map = np.full((10, 10), 0, dtype=float)
        mask = np.full((10, 10), False, dtype=bool)
        numpy_s = np.array(self.state_row, dtype=object)
        s_s = numpy_s.reshape((10, 10))  # reshape of state space

        for x in range(10):
            for y in range(10):
                if s_s[x][y] in str(self.holes_co) or s_s[x][y] == s_s[9][9]:
                    mask[x][y] = True
                elif s_s[x][y] in Q.index:
                    h_map[x][y] = Q.loc[s_s[x][y], :].max()

        plt.figure(1)
        sns.heatmap(h_map, annot=True, cmap='RdBu_r', square=True, mask=mask, linewidths=0.3, linecolor='black', annot_kws={'size': 10})  # plot the heatmap with mask
        plt.title('Optimal state-action value q*(s,a)-({})'.format(name))
        # plt.show()


if __name__ == '__main__':
    env = Lake()
    env.mainloop()
