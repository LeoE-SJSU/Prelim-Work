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

# This code is an edited version to the one found by pytorch DQN tutorial.
# To see original please see docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Setup the specific gymnasium program
# See gymnasium.farama.org for more options, which include things like pendulum, or even walking robots.

env = gym.make("CartPole-v1", render_mode='human')

# Set up matplotlib.
# This is useful for plotting the results.
# See w3schools.com/python/matplotlib_pyplot.asp for explanations

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# Tuple is a list that cannot be changed once the data has been input.
# So what we are doing here is creating a group of duples that can be accessed under the word Transition
# So I can input info like Transition(1, 2, 3, 4) and this will input state = 1,action = 2, next_state = 3... and so on.
# I can then use the data. Transition.state will output all of state.

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# This class saves data up to a point as "memory". We can reference this information so that we do not have to only
# reference the exact preceding state, but other random states beforehand. This improves DQN training.
class ReplayMemory(object):

    # __init__ is used to initialize objects of this class.
    def __init__(self, capacity):
        # deque creates a queue where the first items get eliminated as new items are added at the end, once maxlen is
        # reached.
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# This class explains the Q-network.
# It tries to predict the expected return after it inputs something specific. It is doing this without doing the physics
# math, but referencing the previous data on how the system behaved.

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

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()

# # Render the initial State !! Added this piece of code
env.render()

n_observations = len(state)

# This initializes the torch.nn.Module onto the device, and it'll get updated later on.
# This is basically loading the module weights.

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

# Initializes the # of steps.
steps_done = 0

# Select action simply decides whether to choose a random action or an action from our trained model

def select_action(state):
    global steps_done
    sample = random.random()
    # It eventually stops selecting random actions and selects only from the model.
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    # Updates # of steps
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# Initializes how long each episode has lasted.
episode_durations = []

# This function plots the results
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    # Change the title of the plot at the end
    if show_result:
        plt.clf()
        #plt.title('DQN agent on Cartpole ')
    # Change title of plot while it is running
    else:
        plt.clf()
        #plt.title('Training...')

    # Title of the labels
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy(), marker = 'o', label = 'Individual episode')
    # Save the duration values
    data_to_save = np.column_stack(durations_t.numpy())
    np.savetxt('training_data.csv', data_to_save)


    # Take 100 episode averages and plot them too
    # This is the orange line
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1) # Change size back to 100
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label = '100 episode average')
        plt.legend(loc = 'lower center')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# This function performs a single step of optimization. This requires a more in-depth explanation found on the pytorch
# tutorial page. But it basically does some math with the tensors to get the loss.

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Define the number of steps we want to run if CUDA is available. Which in the Jetson Orin Nano is yes.

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600 # Tutorial runs up to 600.I want a nice plateau to know it has been trained.
else:
    num_episodes = 20


for i_episode in range(num_episodes):
    # Initialize the environment and get its state.
    # It resets the environment every time. This state varies by model.
    state, info = env.reset()
    # Defines a new state as the tensor including all the og state info plus data type and the device used to run it
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # Choose the next action based on the current state. Initially a random action, eventually an action based on
        # model
        action = select_action(state)

        # env.step() updates the environment with actions returning the next agent observation, reward, and checks
        # whether it has been terminated or truncated.
        observation, reward, terminated, truncated, _ = env.step(action.item())

        # render new visual of current actuator and pole position
        env.render() # ADDED THIS PIECE OF CODE

        # Update reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        # Check if system is done running
        if terminated:
            next_state = None
        else:
            # Create new tensor for the next state
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

#torch.save(policy_net.state_dict(), 'policy_net.pth')
#torch.save(target_net.state_dict(), 'target_net.pth')


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()


