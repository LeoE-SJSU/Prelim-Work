import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1", render_mode='human')
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # nn.Linear applies linear transformation to the incoming data matrix. This important so we can do math with it.
        # in_features is the size of each input sample
        # out_features is the size of each output sample
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    # F.relu applies rectified linear unit function. It outputs the input if it is positive, or outputs zero otherwise.
    # It introduces non-linearity into the system. relu = max(0, x).
    # Basically if the weighted sum is negative, it will not output that. It will instead pass activated output as 0,
    # which is a fast way of telling the system that the output was not an improvement.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    # Change the title of the plot at the end
    if show_result:
        plt.clf()
        plt.title('Result')
    # Change title of plot while it is running
    else:
        plt.clf()
        plt.title('Training...')

    # Title of the labels
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), 'o', label = 'Single episode duration')

    # Save the duration values
    data_to_save = np.column_stack(durations_t.numpy())
    np.savetxt('plot_data.csv', data_to_save)

    # Take 100 episode averages and plot them too
    # This is the orange line
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1) # Change size back to 100
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label = '100 episode average')
        plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

print(n_actions)
print(n_observations)

# This initializes the torch.nn.Module onto the device, and it'll get updated later on.
# This is basically loading the module weights.
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load('target_net.pth'))
model.eval()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 20 # Tutorial runs up to 600.I want a nice plateau to know it has been trained.
else:
    num_episodes = 20

for i_episode in range(num_episodes):

    observation, info = env.reset()
    total_reward = 0
    # done = False

    for t in count():

        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_probs = model(obs_tensor)
            action = torch.argmax(action_probs).item()

        observation, reward, done, truncated, info = env.step(action)

        total_reward += total_reward

        env.render()

        if done or truncated:
            print(f"Episode finished with total reward: {total_reward}")
            episode_durations.append(t + 1)
            plot_durations()
            break

print(model)
print('Complete')

plot_durations(show_result=True)
plt.ioff()
plt.show()