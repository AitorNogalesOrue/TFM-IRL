import pickle
import torch
from model import Actor, Critic, Discriminator
import numpy as np
from torchsummary import summary
#from torchinfo import summary


num_inputs=8
num_actions=2
hidden_size=64
expert_demo = pickle.load(open('expert_demo\expert_demo_300_ms244.665498101087.p', "rb"))
demonstrations = np.array(expert_demo)

discrim = Discriminator(num_inputs + num_actions, hidden_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(num_inputs, num_actions,hidden_size).to(device)
critic = Critic(num_inputs, hidden_size)
print('Discriminator')
#print(critic)
#print(discrim)
summary(discrim,(1, 1, 10))
#print(demonstrations)
#print("demonstrations.shape", demonstrations.shape)
#print(demonstrations[1])