import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        '''This class contains the architecture of the neural network to be used
        ----------------------------
        Parameters : num_inputs -> dimensions of the input data
                     num_outputs -> dimensions of the output data
                     fc1,fc2,fc3 -> fully connected layers of the neural network
        ------------------------------
        '''
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                #Performing Xavier Initialisation
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        '''This function does the forward propogation by using the relu activation function over layers fc1 and fc2 and none on fc3'''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        qvalue = self.fc3(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        # Q- values of current and next 
        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)


        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        # DQN Algorithm 
        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        # Choosing the action as the one with the highest q - value  
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]
