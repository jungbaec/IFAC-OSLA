import numpy as np
import copy
from tqdm import tqdm
from numpy.core.numeric import Inf
import math
import matplotlib.pyplot as plt
from scipy import special as sp
import torch
import torch.nn as nn
import random
from collections import namedtuple
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pickle

############### Parameters #####################
Tb = 10 # chip length per symbol for evaluation
gamma_b_len = 21
gamma_b_grid = np.logspace(-0.2,1.0,gamma_b_len) # initial gamma_b in training
cons_pt = np.array([1, -1]) # BPSK

# Set seed number
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# training
n_train = 1000
bit_len_train = 10 # 5

# testing
bit_len_test = 20 # 5, 50
EbN0dB = range(7) # [0,1,2,3,4,5] # Energy per bit to Noise density Ratio in dB scale, initial gamma_b in testing
Nsim = 100000 # 1000

# noise for policy(target policty smoothing)
policy_noise = 0.2
noise_clip = 0.5

###################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)


def txOneBit(L, Trem): # Trem : remaning time, L : threshold(action), function which gives the LLR, t, error, cross entropy value
  # set threshold to L, transmit one bit
  # use for evaluation(testing)
  global cons_pt, sigma_n
  b = np.random.randint(0,2)
  t = 0
  Lc = 0
  LLR = 0
  while t < Trem:
    t += 1
    noise = sigma_n * np.random.randn()
    y_sym = 1 - 2*b + noise
    w = -1 / (2 * sigma_n**2) * (abs(y_sym-cons_pt) ** 2) # w : llr(y|b = 0)
    Lc += w
    LLR = Lc[0] - Lc[1]
    if abs(LLR) > L or (t == Trem): # Added condition to stop when t == Trem
      break

  if LLR > 0:
    b_hat = 0
  else:
    b_hat = 1

  err = abs(b - b_hat)
  # CE = -( (1-b)*np.log(1/(np.exp(-LLR)+1)) + b*np.log(1/(np.exp(LLR)+1)) )

  return LLR, t, err


def txOneBit_gamma(L, gamma_b, bitRem): # bitRem : remaning bit, function which gives the reward, next states
  # observe in gamma
  # use for RL training
  # may also be replaced by using an analytical function of T_sym(for BPSK)
  T = 100 # resolution
  Eb = 1
  N0 = Eb/gamma_b*T

  t = 0
  LLR = 0
  while t < bitRem*T:
    t += 1
    y = 1 + np.sqrt(N0/2) * np.random.randn()
    LLR += 4*y/N0

    if abs(LLR) > L:
      break

  gamma_b_prime = gamma_b*bitRem * (bitRem*T-t)/(bitRem*T) / (bitRem-1) # give next gamma_b, gamma_b = P*T_rem/N0 = Trem*(Eb/N0) / (Tb*bitRem)
  reward = 1/(np.exp(abs(LLR))+1) 
  # reward = 1/(np.exp(L.detach().numpy()[0][0])+1)

  return reward, gamma_b_prime, bitRem-1


def txOneBit_gamma_coded_trans(L, gamma_b, bitRem): # bitRem : remaning bit, function which gives the reward, next states
  # observe in gamma
  # use for RL training, coded transmission
  T = 100 # resolution
  Eb = 1
  N0 = Eb/gamma_b*T
  
  global cons_pt
  b = np.random.randint(0,2) # b = 0,1 random
  t = 0
  Lc = 0
  LLR = 0
  while t < bitRem*T:
    t += 1
    y = 1 - 2*b + np.sqrt(N0/2) * np.random.randn()
    w = -1 / N0 * (abs(y - cons_pt) ** 2)
    Lc += w # Lc is now 2-element array
    LLR = Lc[0] - Lc[1]

    if abs(LLR) > L:
      break
    
  if LLR > 0:
    b_hat = 0
  else:
    b_hat = 1

  gamma_b_prime = gamma_b*bitRem * (bitRem*T-t)/(bitRem*T) / (bitRem-1) # give next gamma_b, gamma_b = P*T_rem/N0 = Trem*(Eb/N0) / (Tb*bitRem)
  reward = abs(b - b_hat)
  # reward = 1/(np.exp(L.detach().numpy()[0][0])+1)

  return reward, gamma_b_prime, bitRem-1


def qfunc(x):
    return torch.maximum(0.5-0.5*torch.erf(x/torch.sqrt(torch.tensor(2,dtype=torch.float, device=device))),torch.tensor(1e-20,dtype=torch.float, device=device))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.mu = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,8), nn.ReLU(), nn.Linear(8,1))  # state_dim = 2, action_dim = 1, think mu as a actor class, nn.ReLU(), nn.Linear(8,8),
    
  def forward(self, state):
    a = torch.square(self.mu(state))  # a = self.mu(state), a = nn.ReLU(self.mu(state))
    
    return a
  
class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.q1 = nn.Sequential(nn.Linear(3,8), nn.ReLU(), nn.Linear(8,8), nn.ReLU(), nn.Linear(8,1))  # state_dim + action_dim = 3
    self.q2 = nn.Sequential(nn.Linear(3,8), nn.ReLU(), nn.Linear(8,8), nn.ReLU(), nn.Linear(8,1))
    # Deleted the last sigmoid activation (or we can use negative sigmoid as last activation) 
    # self.V = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,1))
    # self.L = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,1))
    # self.sigmoid = nn.Sigmoid()
    self.logsigmoid = nn.LogSigmoid() # always give negative value

  def forward(self, x):
    q1 = self.q1(x)
    q1 = self.logsigmoid(q1)
    q2 = self.q2(x)
    q2 = self.logsigmoid(q2)
    
    return q1, q2

  def Q1(self, x):
    q1 = self.q1(x)
    q1 = self.logsigmoid(q1)
    
    return q1
  
  def print_vals(self, x):
    q1 = self.q1(x)
    q1 = self.logsigmoid(q1)
    q2 = self.q2(x)
    q2 = self.logsigmoid(q2)

    print("Q1=",q1.data)
    print("Q2=",q2.data)
    print("----------")

def select_action(state, actor):
  a = actor(state) + 0.3 * state[:,0] * torch.randn_like(actor(state)).clamp(-noise_clip, noise_clip)  # select action using actor, noise add(exploration)
  return a

def optimize_model(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, memory, batch_size, device, iter):

  if len(memory) < batch_size:
    batch_size = len(memory)

  # sample from replay buffer
  transitions = memory.sample(batch_size)
  Transition = namedtuple('Transition',
                          ('state', 'action', 'next_state', 'reward'))
  batch = Transition(*zip(*transitions))

  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  next_state_batch = torch.cat(batch.next_state)
  non_final_mask = torch.tensor(tuple(map(lambda s: s>1, next_state_batch[:,1])), device=device, dtype=torch.bool)  
  final_mask = torch.tensor(tuple(map(lambda s: s==1, next_state_batch[:,1])), device=device, dtype=torch.bool)    

  x_batch = torch.cat((action_batch, state_batch),1)
  gamma_b_prime = next_state_batch[final_mask,0]
  
  with torch.no_grad():
    # Select action according to policy and add clipped noise
    next_action_batch = actor_target(next_state_batch) + (torch.randn_like(action_batch) * policy_noise).clamp(-noise_clip, noise_clip)
    next_x_batch = torch.cat((next_action_batch, next_state_batch),1)
    
    # Compute the target Q value
    target_Q = reward_batch # getting negative reward
    target_Q[final_mask] += 0.99 * torch.unsqueeze(-qfunc(torch.sqrt(2*gamma_b_prime)),1) # discount factor = 1, but can add 0.99 to made stable
    target_Q1, target_Q2 = critic_target(next_x_batch)
    target_Q[non_final_mask] += 0.99 * torch.min(target_Q1[non_final_mask], target_Q2[non_final_mask]) # discount factor = 1, but can add 0.99 to made stable
    
  # Get current Q estimates
  current_Q1, current_Q2 = critic(x_batch)
  
  # Compute critic loss
  critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
  critic_loss_array.append(critic_loss.item())

	# Optimize the critic
  critic_optimizer.zero_grad()
  critic_loss.backward()
  critic_optimizer.step()
   
  # Delayed policy updates
  if iter % 2 == 0:
    # Compute actor loss
    x_batch_prime = torch.cat((actor(state_batch), state_batch),1)
    actor_loss = -critic.Q1(x_batch_prime).mean()
    actor_loss_array.append(actor_loss.item())
    
    # Optimize the actor 
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Update the frozen target models
    tau=0.05
    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
      
    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
# Training algorithm
def train(episodes, *args):

  if not args:
    bit_len = 1
  else:
    bit_len = args[0]
    if len(args) > 1:
      actor = args[1]
    else:
      actor = Actor().to(device)
      # actor.load_state_dict(torch.load("actor_1_weights.pth")) # trained in 10 bit case
      
  actor_target = copy.deepcopy(actor)
  critic = Critic().to(device)
  # critic.load_state_dict(torch.load("critic_1_weights.pth")) # trained in 10 bit case
  critic_target = copy.deepcopy(critic)
  
  memory = ReplayMemory(20000)     # Memory(Capacity) for Experience Replay
  actor_optimizer = optim.Adam(actor.parameters(), lr=3e-5) # Can tune the learning rate smaller one(3e-4, 1e-4, 3e-5...)
  critic_optimizer = optim.Adam(critic.parameters(), lr=3e-5)

  Iterations = 20 # 20
  batch_size = 128
  
  for i in range(bit_len-1,0,-1): # bit_len-1
    print("==================")
    print("training bit",i)

    for episode in tqdm(range(episodes)):

      ## single random rollout
      gamma_b = np.random.choice(gamma_b_grid) # random starting gamma_b
      bitRem = torch.tensor([[bit_len - i + 1]], dtype=torch.float, device=device) # bit remaining BEFORE transmission
      state = torch.tensor([[gamma_b, bitRem]], dtype=torch.float, device=device)

      if (bitRem == 1):
        reward = torch.tensor([[-qfunc(np.sqrt(2*gamma_b))]], dtype=torch.float, device=device) # reward as negative BER
        # changed sign of reward
        gamma_b_prime = 0.
        bitRem -= 1
        Lthres = torch.tensor([[select_action(state, actor)]], dtype=torch.float, device=device)
      else:
        Lthres = torch.tensor([[select_action(state, actor)]], dtype=torch.float, device=device)
        reward, gamma_b_prime, bitRem = txOneBit_gamma(Lthres, gamma_b, bitRem)
        # reward, gamma_b_prime, bitRem = txOneBit_gamma_coded_trans(Lthres, gamma_b, bitRem)
        reward = torch.tensor([[reward]], dtype=torch.float, device=device)
        reward = -reward # changed sign of reward
        

      state_prime = torch.tensor([[gamma_b_prime, bitRem]], dtype=torch.float, device=device)

      memory.push(state, Lthres, state_prime, reward)
      episode_reward = reward.item()
      reward_array.append(episode_reward)
      # print(f"Episode Num: {episode} Reward: {episode_reward:.20f}")

      ## optimization loop
      for iter in range(1, Iterations+1):
        optimize_model(actor=actor, critic = critic, actor_target = actor_target, critic_target = critic_target, actor_optimizer=actor_optimizer, 
                       critic_optimizer=critic_optimizer, memory=memory, batch_size=batch_size, device=device, iter=iter) 
        
    
  # END bit len loop
  print("==============")

  return actor, critic


######################### Training #########################
critic_loss_array = []
actor_loss_array = []
reward_array = []

print("#####################")
print("Start actor-critic training...")
actor, critic = train(n_train,bit_len_train)

plt.figure(2)
plt.plot(critic_loss_array)
plt.xlabel('Episodes')
plt.ylabel('Critic_loss')

plt.figure(3)
plt.plot(actor_loss_array)
plt.xlabel('Episodes')
plt.ylabel('Actor_loss')

plt.figure(4)
plt.plot(reward_array)
plt.xlabel('Episodes')
plt.ylabel('Rewards')


####################### Testing ############################
#torch.save(actor.state_dict(), "actor_1_weights.pth")
#torch.save(critic.state_dict(), "critic_1_weights.pth")
actor = Actor().to(device)
critic = Critic().to(device)
actor.load_state_dict(torch.load("actor_1_weights.pth"))
critic.load_state_dict(torch.load("critic_1_weights.pth"))

## Plot what's action looks like
gamma_b = 2
Ltry = np.linspace(0,8,80) * gamma_b
bit_len_plot = 2
for L in Ltry:
  x = torch.tensor([[L,gamma_b,bit_len_plot]],dtype=torch.float)

critic.print_vals(x)

# plt.semilogy(Ltry,Q_array)
# plt.plot(Ltry, Q_array2)
# plt.grid()
# plt.axis([0,16,1e-2,1])
# print(Q_trained(x))


ber = []
Lthres = []
for i in range(1,bit_len_test+1):
  ber = []
  Lthres = []
  for s in gamma_b_grid:
    a = actor(torch.tensor([[s,i * bit_len_train / bit_len_test]],dtype=torch.float))
    Lthres.append(a.detach().numpy()[0][0])
  # print(ber)

  plt.figure(1)
  plt.plot(gamma_b_grid,Lthres,'-o', label = f"bit remaining {i}")

plt.figure(1)
plt.xlabel('Eb/N0 (linear) for Actor-critic-1')
plt.ylabel('Taken action (Lthres) for Actor-critic-1')
plt.title('test bit length 20, bit_rem scaled')
#plt.legend()

plt.show()

## Evaluate BER
ber_ac_1 = []
ber_for_each_bit_ac_1 = np.zeros([bit_len_test, len(EbN0dB)])
Lthres_ac_1 = torch.zeros([bit_len_test, len(EbN0dB)])

for s in EbN0dB:
  # with tf.device('/device:GPU:0'):
  sigma_n = np.sqrt(10**(-s/10)*Tb/2)
  EbN0 = 10 ** (s/10)
  err = 0
  T_mean = 0
  print("EbN0dB=",s)

  for n in tqdm(range(1,Nsim+1)):
    t_sym = np.zeros(bit_len_test)
    Trem = bit_len_test*Tb
    bitRem = bit_len_test
    gamma_b = Trem/Tb*EbN0/bitRem
    for i in range(bit_len_test):
      # decide L: agent
      # action: a = actor(s) 
      states = torch.tensor([[gamma_b, min([bitRem, bit_len_train])]])
      # min([bitRem, bit_len_train])(threshold case), bitRem * bit_len_train / bit_len_test(scaling case)
      Lt = actor(states)
      if n == 1:
        Lthres_ac_1[bitRem - 1, s] = Lt
      # Lt = gamma_b * a_grid[a_idx]
      if bitRem == 1:
        Lt = Inf
        if n == 1:
          Lthres_ac_1[bitRem - 1, s] = Lt

      # transmit one bit: environment
      # new state: Q(s'|s,a)
      LLR, t, er = txOneBit(Lt, Trem)
      # ber_for_each_bit_ac_1[bitRem - 1, s] += er 
      Trem = Trem - t
      bitRem -= 1
      T_mean += t
      err += er
      ber_for_each_bit_ac_1[i, s] += er 
      
      if bitRem:
        gamma_b = Trem/Tb*EbN0/bitRem
    
  
    if err >= 100:
      break
  
  
    if np.min(ber_for_each_bit_ac_1[:, s]) >= 100:
      break
  
  ber_ac_1.append(err / (n * bit_len_test))
  T_mean /= (n * bit_len_test)
  ber_for_each_bit_ac_1[:, s] /= n
  print(err,n,"\n")
  print("================\n")

print("BER =", ber_ac_1)
print("BER for each bit =", ber_for_each_bit_ac_1)
print("Sum of BER for each bit", np.sum(ber_for_each_bit_ac_1, axis = 0))

plt.figure(2)
plt.semilogy(EbN0dB, ber_ac_1, '-o', label="Actor-critic-1, simulation")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")
# MDP_solution_2 = [0.0668, 0.0419, 0.027, 0.0145, 0.0052, 0.0022, 0.00089, 0.00013, 2e-5]
# MDP_solution_5 = [0.0451, 0.0254, 0.0142, 0.0076, 0.002, 0.000594, 0.000142, 0.000018, 8e-7]
# plt.plot([0,1,2,3,4,5,6,7,8],MDP_solution_2, label="MDP, k=2")
# plt.plot([0,1,2,3,4,5,6,7,8],MDP_solution_5, '--',label="MDP, k=5")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(3)
for i in range(1, bit_len_test + 1):
  plt.semilogy(EbN0dB, ber_for_each_bit_ac_1[i-1], '-o', label = f"BER for bit {i}")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER for each bit")
plt.title("EbN0dB Vs BER for each bit")

plt.legend()

plt.figure(4)
for s in EbN0dB:
  plt.semilogy(range(1, bit_len_test + 1), ber_for_each_bit_ac_1[:, s], '-o', label = f"BER for Eb / N0 (dB) {s}")
plt.xlabel("bit")
plt.ylabel("BER for each Eb / N0 (dB)")
plt.title("bit Vs BER for each Eb / N0 (dB)")

plt.legend()

plt.show()

'''
# Save list data
with open("ber_ac_1_bit_len_train_and_test_20_original.pkl","wb") as f: # _len_test_50
    pickle.dump(ber_ac_1, f)

# Save torch matrix
torch.save(Lthres_ac_1, "Lthres_ac_1_bit_len_train_and_test_20_original.pt") # _len_test_50
'''
