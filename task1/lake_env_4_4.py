import numpy as np
import time
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt


UNIT = 80   # pixels
Lake_H = 4  # grid height
Lake_W = 4  # grid width


class Lake(tk.Tk, object):
    def __init__(self):
        super(Lake, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.state_space = ['[40.0, 40.0]', '[120.0, 40.0]', '[200.0, 40.0]', '[280.0, 40.0]',
                            '[40.0, 120.0]', '[120.0, 120.0]', '[200.0, 120.0]', '[280.0, 120.0]',
                            '[40.0, 200.0]', '[120.0, 200.0]', '[200.0, 200.0]', '[280.0, 200.0]',
                            '[40.0, 280.0]', '[120.0, 280.0]', '[200.0, 280.0]', '[280.0, 280.0]']
        self.title('Lake_4*4')
        self.geometry('320x320')
        self._build_lake()

    def _build_lake(self):
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

        # create origin
        origin = np.array([40, 40])

        # create holes
        self.hole_img = tk.PhotoImage(file='hole.gif')
        self.hole1 = self.canvas.create_image(120, 120, anchor='center', image=self.hole_img)
        self.hole2 = self.canvas.create_image(280, 120, anchor='center', image=self.hole_img)
        self.hole3 = self.canvas.create_image(280, 200, anchor='center', image=self.hole_img)
        self.hole4 = self.canvas.create_image(40, 280, anchor='center', image=self.hole_img)

        # create frisbee
        self.frisbee_img = tk.PhotoImage(file='frisbee.gif')
        self.frisbee = self.canvas.create_image(280, 280, anchor='center', image=self.frisbee_img)

        # create robot
        self.robot_img = tk.PhotoImage(file='robot.gif')
        self.robot = self.canvas.create_image(40, 40, anchor='center', image=self.robot_img)

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        # time.sleep(0.1)
        self.canvas.delete(self.robot)
        self.robot_img = tk.PhotoImage(file='robot.gif')
        self.robot = self.canvas.create_image(40, 40, anchor='center', image=self.robot_img)
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
        elif s_ in [self.canvas.coords(self.hole1), self.canvas.coords(self.hole2), self.canvas.coords(self.hole3), self.canvas.coords(self.hole4)]:
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
        h_map = np.full((4, 4), 0, dtype=float)
        mask = np.full((4, 4), False, dtype=bool)
        s_s = [['[40.0, 40.0]', '[120.0, 40.0]', '[200.0, 40.0]', '[280.0, 40.0]'],
               ['[40.0, 120.0]', 'hole', '[200.0, 120.0]', 'hole'],
               ['[40.0, 200.0]', '[120.0, 200.0]', '[200.0, 200.0]', 'hole'],
               ['hole', '[120.0, 280.0]', '[200.0, 280.0]', 'frisbee']]

        for x in range(4):
            for y in range(4):
                if s_s[x][y] == 'hole' or s_s[x][y] == 'frisbee':
                    mask[x][y] = True
                else:
                    h_map[x][y] = Q.loc[s_s[x][y], :].max()

        plt.figure(1)
        sns.heatmap(h_map, annot=True, cmap='RdBu_r', square=True, mask=mask, linewidths=0.3, linecolor='black', annot_kws={'size': 10})  # plot the heatmap with mask
        plt.title('Optimal state-action value q*(s,a)-({})'.format(name))
        # plt.show()


if __name__ == '__main__':
    env = Lake()
    env.mainloop()
