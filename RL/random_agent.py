import numpy as np
import time
from utils import * 

from env import GridWorld 

class RandomAgent():
    def __init__(self, actions):
        self.actions = actions
                           
    def choose_action(self):
        return np.random.choice(self.actions)


if __name__=="__main__":    
    env = GridWorld(shape = np.array((5,5)), obstacles = np.array([[0,1], [1,1], [2,1], [3,1],\
                    [1,3],[2,3],[3,3],[4,3] ]))
    
    state = env.reset()
    actions = env.action.action_idxs
    agent = RandomAgent(actions)
    steps = 0
    while True: 
        steps += 1
        clear()
        action = agent.choose_action()
        state, _, done, _ = env.step(action)
        env.render()
        if done:
            print(f"the agent reached terminal state in {steps} steps")
            break

        time.sleep(.3)
