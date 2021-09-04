import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Memory(object):
    def __init__(self, capacity):
        '''Initialise the class :
        
        ----------------------------
        
        Parameters : capacity -> represents the available space in the Replay Memory
        
        Attributes : memory -> a circular queue of size capacity which stores the namedTuple of Transitions 

        --------------------------------
        '''

        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        '''
        Function to store a transition  represented by the namedTuple Transitions

        '''
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self, batch_size):
        '''Samples k transitions randomly from the memory'''
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        '''Returns how full the replay memory is '''
        return len(self.memory)
