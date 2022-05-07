import random
import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from typing import Callable
#env = gym.make('MountainCar-v0')
#state_dims = env.observation_space.shape[0]
#num_actions = env.action_space.n
#print(f"MountainCar env: State Dimensions: {state_dims}, Number of actions:{num_actions}")


class PreprocessEnv(gym.Wrapper):
    """
    This class is needed to convert betwen values the pytorch understands (i.e.
    tensors) and values the the gym environements can process (e.g., numpy or
    python default values) 
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    # wraps env.reset
    def reset(self):
        state = self.env.reset()
        # convert a numpy value to tensor
        # add a dimension to allow for batch expecution
        # make it a float 
        return torch.from_numpy(state).unsqueeze(dim=0).float()

    def step(self, action):
        action = action.item() # get the value of a tensor 
        next_s, r, done, info = self.env.step(action)
        next_s = torch.from_numpy(next_s).unsqueeze(dim=0).float()
        r = torch.tensor(r).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)
        return next_s, r, done, info

class ReplayMemory:
    def __init__(self, capacity=int(1e6) ):
        self.capacity = capacity
        self.memory =[]
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size)

        # what the next two lines do
        """
        s   a   r   done    s_
        s   a   r   done    s_
        """
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10
    
    def __len__(self):
        return len(self.memory)

class DeepSarsa:
    def __init__(self, env, q_net,  episodes, alpha=0.001, batch_size=32, gamma=.99, epsilon=.05):
        self.q_net = q_net
        self.episodes = episodes
        self.alpha = alpha
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.num_actions = env.action_space.n
        self.batch_size = batch_size
        self.target_q_net = copy.deepcopy(q_net)
        self.target_q_net.eval()

    def policy(self, state):
        if torch.rand(1) < self. epsilon:
            return torch.randint(self.num_actions, (1, 1))
        else:
            #print(f"state:{state}")
            #print(q_net)
            av = self.q_net(state).detach()
            return torch.argmax(av, dim=-1, keepdim=True)
    
    
    def deep_sarsa(self):
        optim = AdamW(self.q_net.parameters(), lr=self.alpha)
        memory = ReplayMemory(capacity= int(1e6))
        stats = {'MSE Loss': [], 'Returns': []}
    
        for episode in tqdm(range(1, self.episodes + 1)):
            state = self.env.reset()
            done = False
            ep_return = 0.
    
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                memory.insert([state, action, reward, done, next_state])
    
                if memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = \
                            memory.sample(self.batch_size)
                    qsa_b = self.q_net(state_b).gather(1, action_b)
                    next_action_b = self.policy(next_state_b)
                    next_qsa_b = self.target_q_net(next_state_b).gather(1, next_action_b)
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
    
                    loss = F.mse_loss(qsa_b, target_b)
                    q_net.zero_grad()
                    loss.backward()
                    optim.step()
    
                    stats['MSE Loss'].append(loss.item())
    
                state = next_state
                ep_return  += reward.item()
    
            stats['Returns'].append(ep_return)
    
            if episode % 10 ==0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
    
        return stats

def test_agent(env: gym.Env, agent, episodes: int = 10) -> None:
    plt.figure(figsize=(8, 8))
    for episode in range(episodes):
        state = env.reset()
        done = False
        img = plt.imshow(env.render(mode='rgb_array'))
        while not done:
            p = agent.policy(state)
            if isinstance(p, np.ndarray):
                action = np.random.choice(4, p=p)
            else:
                action = p
            next_state, _, done, _ = env.step(action)
            img.set_data(env.render(mode='rgb_array'))
            plt.axis('off')
            state = next_state

def plot_stats(stats):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        vals = [np.mean(vals[i-10:i+10]) for i in range(10, len(vals)-10)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_dims = env.observation_space.shape[0]
    num_actions = env.action_space.n
    env = PreprocessEnv(env)
    #state = env.reset()
    #action = torch.tensor(0)
    #next_state, reward, done, _ = env.step(action)
    #print(f"Sample state:{state}")
    #print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")

    q_net = nn.Sequential(
        nn.Linear(state_dims, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions)
    )

    deep_sarsa = DeepSarsa(env, q_net, 150, epsilon=.01)
    plot_stats(deep_sarsa.deep_sarsa())

    test_agent(env,deep_sarsa, 2)
