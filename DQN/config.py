import torch

env_name = 'CartPole-v0' # The environment name 
gamma = 0.99 # The discount rate
batch_size = 32 # batch_size which wud be used during the training process
lr = 0.001 # learning rate
initial_exploration = 1000 # no of steps to randomly explore the environment in order to fill the ReplayMemory with random transitions
goal_score = 200 # the highest possible score which is available , so terminate the 

update_target = 100 # frequency with which the target network should be updates
replay_memory_capacity = 1000 # capacity of the ReplayMemory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = 3000 # no of episodes to train 
