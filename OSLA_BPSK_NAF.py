import numpy as np
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
bit_len_test = 10 # 5, 20, 50
EbN0dB = range(7) # [0,1,2,3,4,5] # Energy per bit to Noise density Ratio in dB scale, initial gamma_b in testing
Nsim = 100000 # 1000

###################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)


def txOneBit(L, Trem):
  # set threshold to L, transmit one bit
  # use for evaluation
  global cons_pt, sigma_n
  b = np.random.randint(0,2) # b = 0,1 random
  t = 0
  Lc = 0
  LLR = 0
  while t < Trem:
    t += 1
    noise = sigma_n * np.random.randn()
    y_sym = 1 - 2*b + noise
    w = -1 / (2 * sigma_n**2) * (abs(y_sym-cons_pt) ** 2) 
    Lc += w # Lc is now 2-element array
    LLR = Lc[0] - Lc[1]
    if abs(LLR) > L or (t == Trem): # Added condition to stop when t == Trem
      break

  if LLR > 0:
    b_hat = 0
  else:
    b_hat = 1

  err = abs(b - b_hat)
  CE = -( (1-b)*np.log(1/(np.exp(-LLR)+1)) + b*np.log(1/(np.exp(LLR)+1)) )

  return LLR, t, err, CE


def txOneBit_gamma(L, gamma_b, bitRem):
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

  gamma_b_prime = gamma_b*bitRem * (bitRem*T-t)/(bitRem*T) / (bitRem-1)
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


class NAF(nn.Module):
  def __init__(self):
    super(NAF, self).__init__()
    self.mu = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,1))
    self.V = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,1))
    self.L = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,1))
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    a = x[:,0:1]
    s = x[:,1:]
    mu = torch.square(self.mu(s))
    A = 1/2 * torch.square(self.L(s)) * torch.square(a - mu) # open toward up quadratic (w.r.t. a), center at mu
    V = self.V(s) # should be negative
    Q = self.sigmoid(A + V) # want to minimize (A+V) -> min Q = sigmoid(V) ~ exp(V)

    return Q, V, mu

  def print_vals(self,x):
    a = x[:,0:1]
    s = x[:,1:]
    mu = torch.square(self.mu(s))
    P = torch.square(self.L(s))
    A = 1/2 * P * torch.square(a - mu) # open toward up quadratic (w.r.t. a), center at mu
    V = self.V(s) # should be negative
    Q = self.sigmoid(A + V) # want to minimize (A+V) -> min Q = sigmoid(V) ~ exp(V)

    print("Q=",Q.data)
    print("V=",V.data)
    print("mu=",mu.data)
    print("P=",P.data)
    print("----------")




def plot_rewards(actual_rewards, n_mean):
    plt.figure()
    plt.clf()
    #expected_rewards_t = torch.tensor(expected_rewards, dtype=torch.float)
    actual_rewards_t = torch.tensor(actual_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    # plt.semilogy(expected_rewards_t.numpy())
    # Take 100 episode averages and plot them too
    # n_mean = 100
    if len(actual_rewards_t) >= n_mean:
      #means_e = expected_rewards_t.unfold(0, n_mean, 1).mean(1).view(-1)
      #means_e = torch.cat((torch.zeros(n_mean-1), means_e))
      #plt.semilogy(means_e.numpy(),label="expected rewards")
      means = actual_rewards_t.unfold(0, n_mean, 1).mean(1).view(-1)
      means = torch.cat((torch.zeros(n_mean-1), means))
      plt.semilogy(means.numpy(),label="actual rewards")
    # plt.savefig(save_path)
    plt.legend()
    plt.show()
    plt.close()



def draw_fitting(policy_net, target_net,memory, batch_size, device):
  if len(memory) < batch_size:
    batch_size = len(memory)

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
  next_x_batch = torch.cat((action_batch, next_state_batch),1)
  
  Q, _, _ = policy_net(x_batch)
  Q_prime, V_prime, _ = target_net(next_x_batch)

  y_target = reward_batch
  gamma_b_prime = next_state_batch[final_mask,0]
  y_target[final_mask] += torch.unsqueeze(qfunc(np.sqrt(2*gamma_b_prime)),1)
  y_target[non_final_mask] += torch.sigmoid(V_prime[non_final_mask])

  plt.figure()
  plt.plot(action_batch.detach().numpy(), torch.log10(Q).detach().numpy(),'.')
  plt.plot(action_batch.detach().numpy(), torch.log10(y_target).detach().numpy(),'x')
  plt.show()
  plt.close()



def select_action(state, policy_net):
  x = torch.cat((torch.tensor([[0]],dtype=torch.float, device=device),state),1)
  with torch.no_grad():
    _, _, mu = policy_net(x)
    return mu + 0.3*state[:,0]*torch.randn(state.shape[0],1, dtype=torch.float, device=device)


def optimize_model(policy_net, target_net, optimizer, memory, batch_size, device):

  epsilon = 1e-12
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
  next_x_batch = torch.cat((action_batch, next_state_batch),1)
  
  Q, _, _ = policy_net(x_batch)
  Q_target, _, _ = target_net(x_batch)
  Q_prime, V_prime, _ = target_net(next_x_batch)
  
  y_target = reward_batch
  gamma_b_prime = next_state_batch[final_mask,0]
  y_target[final_mask] += torch.unsqueeze(qfunc(torch.sqrt(2*gamma_b_prime)),1)
  y_target[non_final_mask] += torch.sigmoid(V_prime[non_final_mask])
  
  loss = torch.mean( 1/torch.sqrt(Q_target + epsilon) * F.binary_cross_entropy(Q, y_target, reduction='none')) # inverse frequency weighting for imbalance data class
  loss_array.append(loss.detach().cpu().numpy())

  # Optimize the model
  optimizer.zero_grad()
  loss.backward()
  if torch.isnan(loss):
    import pdb; pdb.set_trace()

  for name, param in policy_net.named_parameters():
    if param.requires_grad:
      param.grad.data.clamp_(-1, 1)
      if torch.isnan(torch.sum(param.grad)):
        print(param.grad)
        import pdb; pdb.set_trace()

  optimizer.step()
  for param in policy_net.parameters():
    if torch.isnan(torch.sum(param)):
      print(param.grad)
      import pdb; pdb.set_trace()



# Training algorithm
def Train_Q_backward(episodes, *args):

  if not args:
    bit_len = 1
  else:
    bit_len = args[0]
    if len(args) > 1:
      policy_net = args[1]
    else:
      policy_net = NAF().to(device)

  target_net = NAF().to(device)
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()
  memory = ReplayMemory(10000)     # Memory for Experience Replay
  optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

  Iterations = 10
  batch_size = 100

  expected_rewards = []
  actual_rewards = []
  Lthres_array = []
  
  for i in range(bit_len-1,0,-1):
    print("==================")
    print("training bit",i)

    for episode in tqdm(range(episodes)):

      ## single random rollout
      gamma_b = np.random.choice(gamma_b_grid) # random starting gamma_b
      bitRem = torch.tensor([[bit_len - i + 1]], dtype=torch.float, device=device) # bit remaining BEFORE transmission
      state = torch.tensor([[gamma_b, bitRem]], dtype=torch.float, device=device)

      if (bitRem == 1):
        reward = torch.clamp(torch.tensor([[qfunc(np.sqrt(2*gamma_b))]], dtype=torch.float, device=device),min=1e-20) # reward as BER
        gamma_b_prime = 0.
        bitRem -= 1
        Lthres = torch.tensor([[select_action(state, policy_net)]], dtype=torch.float, device=device)
      else:
        Lthres = torch.tensor([[select_action(state, policy_net)]], dtype=torch.float, device=device)
        reward, gamma_b_prime, bitRem = txOneBit_gamma(Lthres, gamma_b, bitRem)
        # reward, gamma_b_prime, bitRem = txOneBit_gamma_coded_trans(Lthres, gamma_b, bitRem)
        reward = torch.clamp(torch.tensor([[reward]], dtype=torch.float, device=device),min=1e-20)


      state_prime = torch.tensor([[gamma_b_prime, bitRem]], dtype=torch.float, device=device)

      memory.push(state, Lthres, state_prime, reward)

      ## optimization loop
      for _ in range(Iterations):
        optimize_model(policy_net=policy_net, target_net=target_net, optimizer=optimizer, memory=memory, batch_size=batch_size, device=device)
        target_net.load_state_dict(policy_net.state_dict())

      # ## plot progress (rewards)
      # if episode % 1 == 0:
      #   gamma_b = 10**0.6
      #   bitRem = bit_len-i+1
      #   # expected
      #   x = torch.tensor([[0, gamma_b, bitRem]], dtype=torch.float)
      #   _,V,Lthres = policy_net(x)
      #   expected_rewards.append(torch.sigmoid(V).detach().numpy()[0][0]/(bit_len-i+1))
      #   Lthres_array.append(Lthres.detach().numpy()[0][0])
      
      # actual
      # policy_net.print_vals(x)
      #reward = 0
      #while bitRem > 1 and gamma_b > 0:
      #  x = torch.tensor([[0, gamma_b, bitRem]], dtype=torch.float)
      #  _,_,Lthres = policy_net(x)
      #  reward_tmp, gamma_b, bitRem = txOneBit_gamma(Lthres, gamma_b, bitRem)
      #  reward += reward_tmp
      # import pdb; pdb.set_trace()
      #reward += qfunc(np.sqrt(2*gamma_b))
      #actual_rewards.append(reward/(bit_len-i+1))


    ## END episodes
    # draw_fitting(policy_net, target_net,memory, len(memory), device)
    
    # # record BER
    # ber = []
    # u = []
    # for s in gamma_b_grid:
    #   _,V,mu = target_net(torch.tensor([[0,s,bit_len-i+1]], dtype=torch.float))
    #   ber.append(torch.sigmoid(V).detach().numpy()[0][0]/(bit_len-i+1))
    #   u.append(mu.detach().numpy()[0][0])
    
  # END bit len loop
  print("==============")
  
  # n_mean = int(episodes/10)
  # plot_rewards(actual_rewards, n_mean)
  
  return target_net


def Train_Q_table_backward(episodes, *args):

  # only for pre-train purpose
  # train a relative simple case

  bit_len = 2
  # dimensions of Q table
  Qd0 = 13
  Qd1 = bit_len
  Qd2 = 21
  gamma_b_grid = np.logspace(-0.2,1.,Qd0)
  a_grid = np.linspace(0,8,Qd2)

  if not args:
    Q_value = np.zeros((Qd0, Qd1, Qd2)) # (gamma_b, bitRem, L)
  else:
    Q_value = args[0].copy()
  TD_sum_array = []
  reward_sum_array = []

  reward_sum = 0
  TD_sum = 0
  epsilon = 0.1
  visitCount = np.zeros((Qd0,Qd1,Qd2))
  w = 0.77

  ## functions
  def get_state_idx(s):
    gamma_b = s[0]
    bitRem = s[1]
    gamma_idx = np.argmin(abs(gamma_b - gamma_b_grid))
    bit_idx = bitRem-1
    return gamma_idx, bit_idx

  def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))


  for i in range(bit_len,0,-1):
    print("training bit",i)
    for episode in tqdm(range(episodes)):
      epsilon = 0.01 + np.exp(-0.01*episode)
      gamma_b = np.random.choice(gamma_b_grid) # random starting gamma_b
      bitRem = bit_len - i + 1 # bit remaining BEFORE transmission
      s = [gamma_b, bitRem]
      old_gamma_idx, old_bit_idx = get_state_idx(s)

      if (bitRem == 1):
        reward = -qfunc(np.sqrt(2*gamma_b))
        gamma_b_prime = 0
        a_idx = -1
        bitRem = 0
      else:
        if np.random.random() < epsilon:
          a_idx = np.random.randint(Qd2)
        else:
          a_idx = np.argmax(Q_value[old_gamma_idx, old_bit_idx, :])
        Lthresh = gamma_b * a_grid[a_idx]
        reward, gamma_b_prime, bitRem = txOneBit_gamma(Lthresh, gamma_b, bitRem)
        reward = -reward

      # bitRem -= 1
      s_prime = [gamma_b_prime, bitRem]
      new_gamma_idx, new_bit_idx = get_state_idx(s_prime)

      visitCount[old_gamma_idx, old_bit_idx, a_idx] += 1
      Q_old = Q_value[old_gamma_idx, old_bit_idx, a_idx] 


      if bitRem == 0:
        TD = reward - Q_old
      elif bitRem == 1:
        TD = reward + Q_value[new_gamma_idx, new_bit_idx,-1] - Q_old
      else:
        TD = reward + np.max(Q_value[new_gamma_idx, new_bit_idx,:]) - Q_old

      learning_rate = 1 / (visitCount[old_gamma_idx, old_bit_idx, a_idx] ** w)
      # learning_rate = 0.02
      Q_value[old_gamma_idx, old_bit_idx, a_idx] += learning_rate*TD # Update Q value

  return Q_value


def pretrain_NAF(Q_table, episodes):
  # fit NAF to a trained Q table

  # produce data
  Qd0, Qd1, Qd2 = Q_table.shape # (gamma_b, bitRem, L)
  gamma_b_grid = np.logspace(-0.2,1.,Qd0)
  a_grid = np.linspace(0,8,Qd2)
  bit_grid = range(1,Qd1+1)

  x_batch = []
  y_target = []
  for i, gamma_b in enumerate(gamma_b_grid):
    for j, bitRem in enumerate(bit_grid[1:],1):
      for k, a in enumerate(a_grid):
        L = a * gamma_b
        Q = Q_table[i,j,k]
        x_batch.append([L, gamma_b, bitRem])
        y_target.append([-Q])

  x_batch = torch.tensor(x_batch, dtype=torch.float, device=device)
  y_target = torch.tensor(y_target, dtype=torch.float, device=device)
  y_target = torch.clamp(y_target, min=0)

  # supervised learning
  policy_net = NAF().to(device)
  optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

  epsilon = 1e-8
  for _ in tqdm(range(episodes)):
    Q_policy,_,_ = policy_net(x_batch)
    loss = torch.mean( 1/torch.sqrt(y_target + epsilon) * F.binary_cross_entropy(Q_policy, y_target, reduction='none'))

    optimizer.zero_grad()
    loss.backward()

    for name, param in policy_net.named_parameters():
      if param.requires_grad:
        param.grad.data.clamp_(-1, 1)

    optimizer.step()

  return policy_net


######################### Training #########################
loss_array = []
'''
print("Tabular Q training...")
Q_table = Train_Q_table_backward(1000)
print("#####################")
print("Pre-train NAF...")
Q_ini = pretrain_NAF(Q_table, 1000)
print("#####################")
print("Start NAF training...")
Q_NAF = Train_Q_backward(n_train,bit_len_train)
# plt.figure(4)
# plt.plot(loss_array)
'''
####################### Testing ############################
#torch.save(Q_NAF.state_dict(), "Q_NAF_weights.pth")
Q_NAF = NAF().to(device)
Q_NAF.load_state_dict(torch.load("Q_NAF_weights.pth")) # trained in 10 bit case

## Plot what's action looks like
gamma_b = 2
Ltry = np.linspace(0,8,80) * gamma_b
Q_array = []
Q_array2 = []
bit_len_plot = 2
for L in Ltry:
  x = torch.tensor([[L,gamma_b,bit_len_plot]],dtype=torch.float)
  Q,V,mu = Q_NAF(x)
  Q_array.append(Q.detach().numpy()[0][0]/bit_len_plot)
  Q_array2.append(torch.log(Q/(1-Q)).detach().numpy()[0][0])
# print(torch.sigmoid(V).detach().numpy(), mu.detach().numpy())

Q_NAF.print_vals(x)

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
    _,V,mu = Q_NAF(torch.tensor([[0,s,i]],dtype=torch.float))
    ber.append(torch.sigmoid(V).detach().numpy()[0][0]/i)
    Lthres.append(mu.detach().numpy()[0][0])
  # print(ber)
  plt.figure(1)
  plt.semilogy(10*np.log10(gamma_b_grid),ber,'-o')

  plt.figure(2)
  plt.plot(gamma_b_grid,Lthres,'-o')

plt.figure(1)
ber_a1 = []
ber_a2 = []
for s in gamma_b_grid:
  ber_a1.append(qfunc(np.sqrt(2*s)))
  ber_a2.append(np.exp(-4*s))

plt.semilogy(10*np.log10(gamma_b_grid),ber_a1,'k-')
plt.semilogy(10*np.log10(gamma_b_grid),ber_a2,'k--')
plt.axis([-2,10,1e-8,1])
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Expected BER from RL')

plt.figure(2)
plt.xlabel('Eb/N0 (linear)')
plt.ylabel('Taken action (Lthres) for RL')


## Evaluate BER
ber_q = []
ber_for_each_bit_q = np.zeros([bit_len_test, len(EbN0dB)])
Lthres_q = torch.zeros([bit_len_test, len(EbN0dB)])
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
      # action: a = f(s) rely on DNN
      states = torch.tensor([[0, gamma_b, bitRem]])  
      # min([bitRem, bit_len_train]), bitRem * bit_len_train / bit_len_test
      _, _, Lt = Q_NAF(states)
      if n == 1:
        Lthres_q[bitRem - 1, s] = Lt
      # Lt = gamma_b * a_grid[a_idx]
      if bitRem == 1:
        Lt = Inf
        if n == 1:
          Lthres_q[bitRem - 1, s] = Lt

      # transmit one bit: environment
      # new state: Q(s'|s,a)
      LLR, t, er, CE = txOneBit(Lt, Trem)
      Trem = Trem - t
      bitRem -= 1
      T_mean += t
      err += er
      ber_for_each_bit_q[i, s] += er 
      
      if bitRem:
        gamma_b = Trem/Tb*EbN0/bitRem
    
    if err >= 100:
      break

  ber_q.append(err / (n * bit_len_test))
  T_mean /= (n * bit_len_test)
  ber_for_each_bit_q[:, s] /= n
  print(err,n,"\n")
  print("================\n")


print("BER =", ber_q)
print("BER for each bit =", ber_for_each_bit_q)
print("Sum of BER for each bit", np.sum(ber_for_each_bit_q, axis = 0))

plt.figure(3)
plt.semilogy(EbN0dB, ber_q, '-o', label="Q-Learning, simulation")
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

plt.figure(4)
for i in range(1, bit_len_test + 1):
  plt.semilogy(EbN0dB, ber_for_each_bit_q[i-1], '-o', label = f"BER for bit remaining {i}")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER for each bit")
plt.title("EbN0dB Vs BER for each bit")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(5)
for s in EbN0dB:
  plt.semilogy(range(1, bit_len_test + 1), ber_for_each_bit_q[:, s], '-o', label = f"BER for Eb / N0 (dB) {s}")
plt.xlabel("bit")
plt.ylabel("BER for each Eb / N0 (dB)")
plt.title("bit Vs BER for each Eb / N0 (dB)")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.show()

'''
# Save list data
with open("ber_q_bit_len_test_10_using_0_1_reward.pkl","wb") as f: # _len_test_50
    pickle.dump(ber_q, f)

# Save torch matrix
torch.save(Lthres_q, "Lthres_q_bit_len_test_10_using_0_1_reward.pt") # _len_test_50
'''

