import numpy as np
import time

from env import GridWorld 
from utils import *

class Sarsa():
    def __init__(self, env, episodes=10000, epsilon=.2, alpha=.1, gamma=.99):
        self.action_values = np.zeros((env.rows, env.cols, env.action.action_n))
        self.episodes =  episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(env.action.action_idxs) 
        else:
            av = self.action_values [state[0]][state[1]]
            return np.random.choice(np.flatnonzero(av == av.max()))

    def sarsa(self):
        for _ in range(1, self.episodes+1):
            state = env.reset()
            action = self.policy(state)
            done = False

            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = self.policy(state)

                qsa = self.action_values[state[0]][state[1]][action] 
                next_qsa = self.action_values[next_state[0]][next_state[1]][next_action] 
                self.action_values[state[0]][state[1]][action] = qsa + self.alpha *(reward + self.gamma * next_qsa - qsa)

                state = next_state
                action = next_action


if __name__ == '__main__':
    env = GridWorld(shape = np.array((5,5)), obstacles = np.array([[0,1], [1,1], [2,1], [3,1],\
                    [1,3],[2,3],[3,3],[4,3] ]))
    
    sarsa = Sarsa(env, episodes = 10000, epsilon=.2)
    sarsa.sarsa()
    steps =0
    done = False
    env.reset()
    while True: 
        steps += 1
        clear()
        state = env.get_agent_pos()
        action = sarsa.policy(state)
        state, _, done, _ = env.step(action)
        env.render()
        if done:
            print(f"the agent reached terminal state in {steps} steps")
            plot_q_table(sarsa.action_values, 5)
            break

        time.sleep(.5)

