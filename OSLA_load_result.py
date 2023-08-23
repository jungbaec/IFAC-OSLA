import numpy as np
import copy
from tqdm import tqdm
from numpy.core.numeric import Inf
import math
from mpl_toolkits.mplot3d import Axes3D
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
import scipy.io

gamma_b_grid = np.logspace(-0.2,1.0,13) # initial gamma_b in training

# Load the saved matlab file
mat_file_1 = scipy.io.loadmat("ber_sim_K_sim_10.mat")
ber_sim_K_sim_10 = mat_file_1["ber_sim"]

mat_file_2 = scipy.io.loadmat("ber_ind_K_sim_10.mat")
ber_ind_K_sim_10 = mat_file_2["ber_ind"]

mat_file_3 = scipy.io.loadmat("ber_sim_K_sim_20.mat")
ber_sim_K_sim_20 = mat_file_3["ber_sim"]

mat_file_4 = scipy.io.loadmat("ber_ind_K_sim_20.mat")
ber_ind_K_sim_20 = mat_file_4["ber_ind"]

mat_file_5 = scipy.io.loadmat("ber_sim_K_sim_50.mat")
ber_sim_K_sim_50 = mat_file_5["ber_sim"]

mat_file_6 = scipy.io.loadmat("action_L_thres_array_K_sim_10.mat")
action_L_thres_array_K_sim_10 = mat_file_6["action_L_thres_array"]

mat_file_7 = scipy.io.loadmat("action_L_thres_array_K_sim_20.mat")
action_L_thres_array_K_sim_20 = mat_file_7["action_L_thres_array"]

mat_file_8 = scipy.io.loadmat("action_L_thres_array_K_sim_50.mat")
action_L_thres_array_K_sim_50 = mat_file_8["action_L_thres_array"]

mat_file_9 = scipy.io.loadmat("g_table_sum_10.mat")
g_table_10 = mat_file_9["g_table"]

mat_file_10 = scipy.io.loadmat("g_table_sum.mat")
g_table_8 = mat_file_10["g_table"]

mat_file_11 = scipy.io.loadmat("g_table_sum_20.mat")
g_table_20 = mat_file_11["g_table"]

mat_file_12 = scipy.io.loadmat("g_table_sum_2.mat")
g_table_2 = mat_file_12["g_table"]

mat_file_13 = scipy.io.loadmat("ber_ind_K_sim_2.mat")
ber_ind_K_sim_2 = mat_file_13["ber_ind"]


# Load the saved list
with open("ber_ac_1_bit_len_test_10.pkl","rb") as f:
    ber_ac_1_len_10 = pickle.load(f)
    
with open("ber_ac_1_bit_len_test_10_using_0_1_reward.pkl","rb") as f:
    ber_ac_1_len_10_use_0_1_reward = pickle.load(f)

with open("ber_ac_2_bit_len_test_10.pkl","rb") as f:
    ber_ac_2_len_10 = pickle.load(f)
    
with open("ber_ac_2_bit_len_test_10_using_0_1_reward.pkl","rb") as f:
    ber_ac_2_len_10_use_0_1_reward = pickle.load(f)
    
with open("ber_q_bit_len_test_10.pkl","rb") as f:
    ber_q_len_10 = pickle.load(f)

with open("ber_ac_1_bit_len_test_20.pkl","rb") as f:
    ber_ac_1_len_20 = pickle.load(f)

with open("ber_ac_1_bit_len_test_20_threshold_bit_rem_10.pkl","rb") as f:
    ber_ac_1_len_20_th_bit_rem_10 = pickle.load(f)

with open("ber_ac_1_bit_len_test_20_threshold_time_rem_10.pkl","rb") as f:
    ber_ac_1_len_20_th_time_rem_10 = pickle.load(f)

with open("ber_ac_1_bit_len_test_20_scale_bit_rem_10.pkl","rb") as f:
    ber_ac_1_len_20_sc_bit_rem_10 = pickle.load(f)
    
with open("ber_ac_1_bit_len_train_and_test_20_original.pkl","rb") as f:
    ber_ac_1_len_20_original = pickle.load(f)
    
with open("ber_ac_1_bit_len_train_and_test_20_init_param_10.pkl","rb") as f:
    ber_ac_1_len_20_init_param_10 = pickle.load(f)
    
with open("ber_ac_1_bit_len_train_and_test_20_init_param_10_train_only_unobserved.pkl","rb") as f:
    ber_ac_1_len_20_init_param_10_train_only_unobserved = pickle.load(f)
    
with open("ber_ac_2_bit_len_test_20.pkl","rb") as f:
    ber_ac_2_len_20 = pickle.load(f)

with open("ber_ac_2_bit_len_test_20_threshold_bit_rem_10.pkl","rb") as f:
    ber_ac_2_len_20_th_bit_rem_10 = pickle.load(f)

with open("ber_ac_2_bit_len_test_20_threshold_time_rem_10.pkl","rb") as f:
    ber_ac_2_len_20_th_time_rem_10 = pickle.load(f)
    
with open("ber_ac_2_bit_len_test_20_scale_bit_rem_10.pkl","rb") as f:
    ber_ac_2_len_20_sc_bit_rem_10 = pickle.load(f)
    
with open("ber_ac_2_bit_len_train_and_test_20_original.pkl","rb") as f:
    ber_ac_2_len_20_original = pickle.load(f)
    
with open("ber_ac_2_bit_len_train_and_test_20_init_param_10.pkl","rb") as f:
    ber_ac_2_len_20_init_param_10 = pickle.load(f)
    
with open("ber_ac_2_bit_len_train_and_test_20_init_param_10_train_only_unobserved.pkl","rb") as f:
    ber_ac_2_len_20_init_param_10_train_only_unobserved = pickle.load(f)
    
with open("ber_q_bit_len_test_20.pkl","rb") as f:
    ber_q_len_20 = pickle.load(f)

with open("ber_q_bit_len_test_20_threshold_bit_rem_10.pkl","rb") as f:
    ber_q_len_20_th_bit_rem_10 = pickle.load(f)

with open("ber_q_bit_len_test_20_threshold_time_rem_10.pkl","rb") as f:
    ber_q_len_20_th_time_rem_10 = pickle.load(f)

with open("ber_q_bit_len_test_20_scale_bit_rem_10.pkl","rb") as f:
    ber_q_len_20_sc_bit_rem_10 = pickle.load(f)

with open("ber_ac_1_bit_len_test_50.pkl","rb") as f:
    ber_ac_1_len_50 = pickle.load(f)

with open("ber_ac_1_bit_len_test_50_threshold_bit_rem_10.pkl","rb") as f:
    ber_ac_1_len_50_th_bit_rem_10 = pickle.load(f)

with open("ber_ac_1_bit_len_test_50_threshold_time_rem_10.pkl","rb") as f:
    ber_ac_1_len_50_th_time_rem_10 = pickle.load(f)

with open("ber_ac_1_bit_len_test_50_scale_bit_rem_10.pkl","rb") as f:
    ber_ac_1_len_50_sc_bit_rem_10 = pickle.load(f)
    
with open("ber_ac_2_bit_len_test_50.pkl","rb") as f:
    ber_ac_2_len_50 = pickle.load(f)

with open("ber_ac_2_bit_len_test_50_threshold_bit_rem_10.pkl","rb") as f:
    ber_ac_2_len_50_th_bit_rem_10 = pickle.load(f)

with open("ber_ac_2_bit_len_test_50_threshold_time_rem_10.pkl","rb") as f:
    ber_ac_2_len_50_th_time_rem_10 = pickle.load(f)
    
with open("ber_ac_2_bit_len_test_50_scale_bit_rem_10.pkl","rb") as f:
    ber_ac_2_len_50_sc_bit_rem_10 = pickle.load(f)
    
with open("ber_q_bit_len_test_50.pkl","rb") as f:
    ber_q_len_50 = pickle.load(f)
    
with open("ber_q_bit_len_test_50_threshold_bit_rem_10.pkl","rb") as f:
    ber_q_len_50_th_bit_rem_10 = pickle.load(f)

with open("ber_q_bit_len_test_50_threshold_time_rem_10.pkl","rb") as f:
    ber_q_len_50_th_time_rem_10 = pickle.load(f)
    
with open("ber_q_bit_len_test_50_scale_bit_rem_10.pkl","rb") as f:
    ber_q_len_50_sc_bit_rem_10 = pickle.load(f)
    
# Load the saved pytorch tensor data

Lthres_ac_1_len_10 = torch.load("Lthres_ac_1_bit_len_test_10.pt")
Lthres_ac_2_len_10 = torch.load("Lthres_ac_2_bit_len_test_10.pt")
Lthres_q_len_10 = torch.load("Lthres_q_bit_len_test_10.pt")

Lthres_ac_1_len_20 = torch.load("Lthres_ac_1_bit_len_test_20.pt")
Lthres_ac_1_len_20_th_bit_rem_10 = torch.load("Lthres_ac_1_bit_len_test_20_threshold_bit_rem_10.pt")
Lthres_ac_1_len_20_th_time_rem_10 = torch.load("Lthres_ac_1_bit_len_test_20_threshold_time_rem_10.pt")
Lthres_ac_1_len_20_sc_bit_rem_10 = torch.load("Lthres_ac_1_bit_len_test_20_scale_bit_rem_10.pt")

Lthres_ac_2_len_20 = torch.load("Lthres_ac_2_bit_len_test_20.pt")
Lthres_ac_2_len_20_th_bit_rem_10 = torch.load("Lthres_ac_2_bit_len_test_20_threshold_bit_rem_10.pt")
Lthres_ac_2_len_20_th_time_rem_10 = torch.load("Lthres_ac_2_bit_len_test_20_threshold_time_rem_10.pt")
Lthres_ac_2_len_20_sc_bit_rem_10 = torch.load("Lthres_ac_2_bit_len_test_20_scale_bit_rem_10.pt")

Lthres_q_len_20 = torch.load("Lthres_q_bit_len_test_20.pt")
Lthres_q_len_20_th_bit_rem_10 = torch.load("Lthres_q_bit_len_test_20_threshold_bit_rem_10.pt")
Lthres_q_len_20_th_time_rem_10 = torch.load("Lthres_q_bit_len_test_20_threshold_time_rem_10.pt")
Lthres_q_len_20_sc_bit_rem_10 = torch.load("Lthres_q_bit_len_test_20_scale_bit_rem_10.pt")

Lthres_ac_1_len_50 = torch.load("Lthres_ac_1_bit_len_test_50.pt")
Lthres_ac_1_len_50_th_bit_rem_10 = torch.load("Lthres_ac_1_bit_len_test_50_threshold_bit_rem_10.pt")
Lthres_ac_1_len_50_th_time_rem_10 = torch.load("Lthres_ac_1_bit_len_test_50_threshold_time_rem_10.pt")
Lthres_ac_1_len_50_sc_bit_rem_10 = torch.load("Lthres_ac_1_bit_len_test_50_scale_bit_rem_10.pt")

Lthres_ac_2_len_50 = torch.load("Lthres_ac_2_bit_len_test_50.pt")
Lthres_ac_2_len_50_th_bit_rem_10 = torch.load("Lthres_ac_2_bit_len_test_50_threshold_bit_rem_10.pt")
Lthres_ac_2_len_50_th_time_rem_10 = torch.load("Lthres_ac_2_bit_len_test_50_threshold_time_rem_10.pt")
Lthres_ac_2_len_50_sc_bit_rem_10 = torch.load("Lthres_ac_2_bit_len_test_50_scale_bit_rem_10.pt")

Lthres_q_len_50 = torch.load("Lthres_q_bit_len_test_50.pt")
Lthres_q_len_50_th_bit_rem_10 = torch.load("Lthres_q_bit_len_test_50_threshold_bit_rem_10.pt")
Lthres_q_len_50_th_time_rem_10 = torch.load("Lthres_q_bit_len_test_50_threshold_time_rem_10.pt")
Lthres_q_len_50_sc_bit_rem_10 = torch.load("Lthres_q_bit_len_test_50_scale_bit_rem_10.pt")

#mat_file_2 = scipy.io.loadmat("g_table_sum.mat")
#g_table = mat_file_2["g_table"]

EbN0dB = range(7)
'''
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1, projection='3d')
x_10 = EbN0dB
x_20 = EbN0dB
x_50 = EbN0dB
y_10 = range(2, 11)
y_20 = range(2, 21)
y_50 = range(2, 51)

x_10, y_10 = np.meshgrid(x_10, y_10)
z_10 = action_L_thres_array_K_sim_10[y_10-1, x_10]
ax.plot_surface(x_10, y_10, z_10, label = "test bit length 10")

x_20, y_20 = np.meshgrid(x_20, y_20)
z_20 = action_L_thres_array_K_sim_20[y_20-1, x_20]
ax.plot_surface(x_20, y_20, z_20, label = "test bit length 20")

x_50, y_50 = np.meshgrid(x_50, y_50)
z_50 = action_L_thres_array_K_sim_50[y_50-1, x_50]
ax.plot_surface(x_50, y_50, z_50, label = "test bit length 50")

ax.set_xlabel('Eb/N0 (linear) for MDP')
ax.set_ylabel('bit remaining for MDP')
ax.set_zlabel('Taken action (Lthres) for MDP')
'''

plt.figure(1)
plt.plot(EbN0dB, action_L_thres_array_K_sim_10[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, action_L_thres_array_K_sim_20[1], label = "test bit length 20, bit remaining 2")
plt.plot(EbN0dB, action_L_thres_array_K_sim_50[1], label = "test bit length 50, bit remaining 2")
plt.plot(EbN0dB, action_L_thres_array_K_sim_10[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, action_L_thres_array_K_sim_20[9], label = "test bit length 20, bit remaining 10")
plt.plot(EbN0dB, action_L_thres_array_K_sim_50[9], label = "test bit length 50, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for MDP')
plt.ylabel('Taken action (Lthres) for MDP')
plt.legend()


plt.figure(2)
plt.subplot(2,2,1)
plt.plot(EbN0dB, Lthres_q_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_20.detach().numpy()[1], label = "test bit length 20, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_50.detach().numpy()[1], label = "test bit length 50, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_q_len_20.detach().numpy()[9], label = "test bit length 20, bit remaining 10")
plt.plot(EbN0dB, Lthres_q_len_50.detach().numpy()[9], label = "test bit length 50, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for NAF-based-Q-Learning')
plt.ylabel('Taken action (Lthres) for NAF-based-Q-Learning')
plt.legend()

plt.subplot(2,2,2)
plt.plot(EbN0dB, Lthres_q_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_20_th_bit_rem_10.detach().numpy()[1], label = "test bit length 20, bit_rem upper bounded, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_50_th_bit_rem_10.detach().numpy()[1], label = "test bit length 50, bit_rem upper bounded, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_q_len_20_th_bit_rem_10.detach().numpy()[9], label = "test bit length 20, bit_rem upper bounded, bit remaining 10")
plt.plot(EbN0dB, Lthres_q_len_50_th_bit_rem_10.detach().numpy()[9], label = "test bit length 50, bit_rem upper bounded, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for NAF-based-Q-Learning')
plt.ylabel('Taken action (Lthres) for NAF-based-Q-Learning')
plt.legend()

plt.subplot(2,2,3)
plt.plot(EbN0dB, Lthres_q_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_20_sc_bit_rem_10.detach().numpy()[1], label = "test bit length 20, bit_rem scaled, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_50_sc_bit_rem_10.detach().numpy()[1], label = "test bit length 50, bit_rem scaled, bit remaining 2")
plt.plot(EbN0dB, Lthres_q_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_q_len_20_sc_bit_rem_10.detach().numpy()[9], label = "test bit length 20, bit_rem scaled, bit remaining 10")
plt.plot(EbN0dB, Lthres_q_len_50_sc_bit_rem_10.detach().numpy()[9], label = "test bit length 50, bit_rem scaled, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for NAF-based-Q-Learning')
plt.ylabel('Taken action (Lthres) for NAF-based-Q-Learning')
plt.legend()

plt.figure(3)
plt.subplot(2,2,1)
plt.plot(EbN0dB, Lthres_ac_1_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_20.detach().numpy()[1], label = "test bit length 20, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_50.detach().numpy()[1], label = "test bit length 50, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_1_len_20.detach().numpy()[9], label = "test bit length 20, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_1_len_50.detach().numpy()[9], label = "test bit length 50, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for Actor-critic-1')
plt.ylabel('Taken action (Lthres) for Actor-critic-1')
plt.legend()

plt.subplot(2,2,2)
plt.plot(EbN0dB, Lthres_ac_1_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_20_th_bit_rem_10.detach().numpy()[1], label = "test bit length 20, bit_rem upper bounded, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_50_th_bit_rem_10.detach().numpy()[1], label = "test bit length 50, bit_rem upper bounded, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_1_len_20_th_bit_rem_10.detach().numpy()[9], label = "test bit length 20, bit_rem upper bounded, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_1_len_50_th_bit_rem_10.detach().numpy()[9], label = "test bit length 50, bit_rem upper bounded, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for Actor-critic-1')
plt.ylabel('Taken action (Lthres) for Actor-critic-1')
plt.legend()

plt.subplot(2,2,3)
plt.plot(EbN0dB, Lthres_ac_1_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_20_sc_bit_rem_10.detach().numpy()[1], label = "test bit length 20, bit_rem scaled, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_50_sc_bit_rem_10.detach().numpy()[1], label = "test bit length 50, bit_rem scaled, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_1_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_1_len_20_sc_bit_rem_10.detach().numpy()[9], label = "test bit length 20, bit_rem scaled, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_1_len_50_sc_bit_rem_10.detach().numpy()[9], label = "test bit length 50, bit_rem scaled, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for Actor-critic-1')
plt.ylabel('Taken action (Lthres) for Actor-critic-1')
plt.legend()

plt.figure(4)
plt.subplot(2,2,1)
plt.plot(EbN0dB, Lthres_ac_2_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_20.detach().numpy()[1], label = "test bit length 20, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_50.detach().numpy()[1], label = "test bit length 50, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_2_len_20.detach().numpy()[9], label = "test bit length 20, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_2_len_50.detach().numpy()[9], label = "test bit length 50, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for Actor-critic-2')
plt.ylabel('Taken action (Lthres) for Actor-critic-2')
plt.legend()

plt.subplot(2,2,2)
plt.plot(EbN0dB, Lthres_ac_2_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_20_th_bit_rem_10.detach().numpy()[1], label = "test bit length 20, bit_rem upper bounded, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_50_th_bit_rem_10.detach().numpy()[1], label = "test bit length 50, bit_rem upper bounded, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_2_len_20_th_bit_rem_10.detach().numpy()[9], label = "test bit length 20, bit_rem upper bounded, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_2_len_50_th_bit_rem_10.detach().numpy()[9], label = "test bit length 50, bit_rem upper bounded, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for Actor-critic-2')
plt.ylabel('Taken action (Lthres) for Actor-critic-2')
plt.legend()

plt.subplot(2,2,3)
plt.plot(EbN0dB, Lthres_ac_2_len_10.detach().numpy()[1], label = "test bit length 10, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_20_sc_bit_rem_10.detach().numpy()[1], label = "test bit length 20, bit_rem scaled, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_50_sc_bit_rem_10.detach().numpy()[1], label = "test bit length 50, bit_rem scaled, bit remaining 2")
plt.plot(EbN0dB, Lthres_ac_2_len_10.detach().numpy()[9], label = "test bit length 10, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_2_len_20_sc_bit_rem_10.detach().numpy()[9], label = "test bit length 20, bit_rem scaled, bit remaining 10")
plt.plot(EbN0dB, Lthres_ac_2_len_50_sc_bit_rem_10.detach().numpy()[9], label = "test bit length 50, bit_rem scaled, bit remaining 10")
plt.xlabel('Eb/N0 (dB) for Actor-critic-2')
plt.ylabel('Taken action (Lthres) for Actor-critic-2')
plt.legend()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

def qfunc(x):
    return torch.maximum(0.5-0.5*torch.erf(x/torch.sqrt(torch.tensor(2,dtype=torch.float, device=device))),torch.tensor(1e-20,dtype=torch.float, device=device))
    
plt.figure(5)
plt.semilogy(EbN0dB, ber_sim_K_sim_10[9], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_10, '-o', label="Actor-critic-2")
plt.semilogy(EbN0dB, ber_ac_1_len_10, '-o', label="Actor-critic-1")
plt.semilogy(EbN0dB, ber_q_len_10, '-o', label="NAF-based-Q-Learning")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 10")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(6)
plt.subplot(2,2,1)
plt.semilogy(EbN0dB, ber_sim_K_sim_20[19], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_20, '-o', label="Actor-critic-2")
plt.semilogy(EbN0dB, ber_ac_1_len_20, '-o', label="Actor-critic-1")
plt.semilogy(EbN0dB, ber_q_len_20, '-o', label="NAF-based-Q-Learning")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 20")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.subplot(2,2,2)
plt.semilogy(EbN0dB, ber_sim_K_sim_20[19], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_20_th_bit_rem_10, '-o', label="Actor-critic-2, bit_rem upper bounded")
plt.semilogy(EbN0dB, ber_ac_1_len_20_th_bit_rem_10, '-o', label="Actor-critic-1, bit_rem upper bounded")
plt.semilogy(EbN0dB, ber_q_len_20_th_bit_rem_10, '-o', label="NAF-based-Q-Learning, bit_rem upper bounded")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 20")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.subplot(2,2,3)
plt.semilogy(EbN0dB, ber_sim_K_sim_20[19], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_20_sc_bit_rem_10, '-o', label="Actor-critic-2, bit_rem scaled")
plt.semilogy(EbN0dB, ber_ac_1_len_20_sc_bit_rem_10, '-o', label="Actor-critic-1, bit_rem scaled")
plt.semilogy(EbN0dB, ber_q_len_20_sc_bit_rem_10, '-o', label="NAF-based-Q-Learning, bit_rem scaled")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 20")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.subplot(2,2,4)
plt.semilogy(EbN0dB, ber_sim_K_sim_20[19], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_20_th_bit_rem_10, '-o', label="Actor-critic-2, bit_rem upper bounded")
plt.semilogy(EbN0dB, ber_ac_1_len_20_th_bit_rem_10, '-o', label="Actor-critic-1, bit_rem upper bounded")
plt.semilogy(EbN0dB, ber_ac_2_len_20_sc_bit_rem_10, '-o', label="Actor-critic-2, bit_rem scaled")
plt.semilogy(EbN0dB, ber_ac_1_len_20_sc_bit_rem_10, '-o', label="Actor-critic-1, bit_rem scaled")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 20")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(7)
plt.subplot(2,2,1)
plt.semilogy(EbN0dB, ber_sim_K_sim_50[49], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_50, '-o', label="Actor-critic-2")
plt.semilogy(EbN0dB, ber_ac_1_len_50, '-o', label="Actor-critic-1")
plt.semilogy(EbN0dB, ber_q_len_50, '-o', label="NAF-based-Q-Learning")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 50")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.subplot(2,2,2)
plt.semilogy(EbN0dB, ber_sim_K_sim_50[49], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_50_th_bit_rem_10, '-o', label="Actor-critic-2, bit_rem upper bounded")
plt.semilogy(EbN0dB, ber_ac_1_len_50_th_bit_rem_10, '-o', label="Actor-critic-1, bit_rem upper bounded")
plt.semilogy(EbN0dB, ber_q_len_50_th_bit_rem_10, '-o', label="NAF-based-Q-Learning, bit_rem upper bounded")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 50")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.subplot(2,2,3)
plt.semilogy(EbN0dB, ber_sim_K_sim_50[49], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_50_sc_bit_rem_10, '-o', label="Actor-critic-2, bit_rem scaled")
plt.semilogy(EbN0dB, ber_ac_1_len_50_sc_bit_rem_10, '-o', label="Actor-critic-1, bit_rem scaled")
plt.semilogy(EbN0dB, ber_q_len_50_sc_bit_rem_10, '-o', label="NAF-based-Q-Learning, bit_rem scaled")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 50")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.subplot(2,2,4)
plt.semilogy(EbN0dB, ber_sim_K_sim_50[49], '-o', label="MDP result")
plt.semilogy(EbN0dB, ber_ac_2_len_50_th_bit_rem_10, '-o', label="Actor-critic-2, bit_rem upper bounded")
plt.semilogy(EbN0dB, ber_ac_1_len_50_th_bit_rem_10, '-o', label="Actor-critic-1, bit_rem upper bounded")
plt.semilogy(EbN0dB, ber_ac_2_len_50_sc_bit_rem_10, '-o', label="Actor-critic-2, bit_rem scaled")
plt.semilogy(EbN0dB, ber_ac_1_len_50_sc_bit_rem_10, '-o', label="Actor-critic-1, bit_rem scaled")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 50")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(8)
plt.semilogy(EbN0dB, ber_ac_1_len_20_original, '-o', label="Actor-critic-1")
plt.semilogy(EbN0dB, ber_ac_1_len_20_init_param_10, '-o', label="Actor-critic-1 with initial param")
plt.semilogy(EbN0dB, ber_ac_1_len_20_init_param_10_train_only_unobserved, '-o', label="Actor-critic-1 with initial param, train for only unobserved bit rem")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for comparing the impact of initialization")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(9)

# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.semilogy(EbN0dB, ber_ac_2_len_20_original, '-o', label="Actor-critic-2")
plt.semilogy(EbN0dB, ber_ac_2_len_20_init_param_10, '-o', label="Actor-critic-2 with initial param")
plt.semilogy(EbN0dB, ber_ac_2_len_20_init_param_10_train_only_unobserved, '-o', label="Actor-critic-2 with initial param, train for only unobserved bit rem")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for comparing the impact of initialization")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(10)
plt.semilogy(EbN0dB, ber_ac_1_len_10, '-o', label="Actor-critic-1")
plt.semilogy(EbN0dB, ber_ac_1_len_10_use_0_1_reward, '-o', label="Actor-critic-1, use 0-1 reward")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 10, using different form of reward")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(11)
plt.semilogy(EbN0dB, ber_ac_2_len_10, '-o', label="Actor-critic-2")
plt.semilogy(EbN0dB, ber_ac_2_len_10_use_0_1_reward, '-o', label="Actor-critic-2, use 0-1 reward")
# plt.semilogy(10*np.log10(gamma_b_grid), ber_a[bit_len_test-1], '--', label="Q-Learning, analysis")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER")
plt.title("EbN0dB Vs BER for bit_len_test = 10, using different form of reward")
EbN0dB_bpsk = np.linspace(0,10,21)
FL_BPSK = qfunc(np.sqrt(2*10**(EbN0dB_bpsk/10)))
OSLA = np.exp(-4*10**(EbN0dB_bpsk/10))
plt.semilogy(EbN0dB_bpsk, FL_BPSK, label = "FL_BPSK")
plt.semilogy(EbN0dB_bpsk, OSLA, '--',label="OSLA lower bound")

plt.legend()
plt.grid()
plt.axis([0,10,1e-6,1])

plt.figure(12)
for i in range(1, 11):
  plt.semilogy(EbN0dB, ber_ind_K_sim_10[i-1], '-o', label = f"BER for bit {i}")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER for each bit")
plt.title("EbN0dB Vs BER for each bit")

plt.legend()
#plt.grid()
#plt.axis([0,10,1e-6,1])

plt.figure(13)
for s in EbN0dB:
  plt.semilogy(range(1, 11), ber_ind_K_sim_10[:, s], '-o', label = f"BER for Eb / N0 (dB) {s}")
plt.xlabel("bit")
plt.ylabel("BER for each Eb / N0 (dB)")
plt.title("bit Vs BER for each Eb / N0 (dB)")

plt.legend()
#plt.grid()
#plt.axis([0,10,1e-6,1])

plt.figure(14)
for i in range(1, 21):
  plt.semilogy(EbN0dB, ber_ind_K_sim_20[i-1], '-o', label = f"BER for bit {i}")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER for each bit")
plt.title("EbN0dB Vs BER for each bit")

plt.legend()
#plt.grid()
#plt.axis([0,10,1e-6,1])

plt.figure(15)
for s in EbN0dB:
  plt.semilogy(range(1, 21), ber_ind_K_sim_20[:, s], '-o', label = f"BER for Eb / N0 (dB) {s}")
plt.xlabel("bit")
plt.ylabel("BER for each Eb / N0 (dB)")
plt.title("bit Vs BER for each Eb / N0 (dB)")

plt.legend()
#plt.grid()
#plt.axis([0,10,1e-6,1])

Lthres = []
for i in range(1, 10):
  ber = []
  Lthres = []
  for s in range(3, 16):
    a = g_table_10[i, s]
    Lthres.append(a)
  # print(ber)

  plt.figure(16)
  plt.plot(gamma_b_grid,Lthres,'-o', label = f"bit remaining {i+1}")

plt.figure(16)
plt.xlabel('Eb/N0 (linear) for MDP')
plt.ylabel('Taken action (Lthres) for MDP')
plt.title('bit length 10')
plt.legend()

Lthres = []
for i in range(1, 8):
  ber = []
  Lthres = []
  for s in range(3, 16):
    a = g_table_8[i, s]
    Lthres.append(a)
  # print(ber)

  plt.figure(17)
  plt.plot(gamma_b_grid,Lthres,'-o', label = f"bit remaining {i+1}")

plt.figure(17)
plt.xlabel('Eb/N0 (linear) for MDP')
plt.ylabel('Taken action (Lthres) for MDP')
plt.title('bit length 8')
plt.legend()

Lthres = []
for i in range(1, 20):
  ber = []
  Lthres = []
  for s in range(3, 16):
    a = g_table_20[i, s]
    Lthres.append(a)
  # print(ber)

  plt.figure(18)
  plt.plot(gamma_b_grid,Lthres,'-o', label = f"bit remaining {i+1}")
  
plt.figure(18)
plt.xlabel('Eb/N0 (linear) for MDP')
plt.ylabel('Taken action (Lthres) for MDP')
plt.title('bit length 20')
plt.legend()
  
Lthres = []
for i in range(1, 2):
  ber = []
  Lthres = []
  for s in range(3, 16):
    a = g_table_2[i, s]
    Lthres.append(a)
  # print(ber)

  plt.figure(19)
  plt.plot(gamma_b_grid,Lthres,'-o', label = f"bit remaining {i+1}")

plt.figure(19)
plt.xlabel('Eb/N0 (linear) for MDP')
plt.ylabel('Taken action (Lthres) for MDP')
plt.title('bit length 2')
plt.legend()

plt.figure(20)
for i in range(1, 3):
  plt.semilogy(EbN0dB, ber_ind_K_sim_2[i-1], '-o', label = f"BER for bit {i}")
plt.xlabel("Eb / N0 (dB)")
plt.ylabel("BER for each bit")
plt.title("EbN0dB Vs BER for each bit")

plt.legend()
#plt.grid()
#plt.axis([0,10,1e-6,1])

plt.figure(21)
for s in EbN0dB:
  plt.semilogy(range(1, 3), ber_ind_K_sim_2[:, s], '-o', label = f"BER for Eb / N0 (dB) {s}")
plt.xlabel("bit")
plt.ylabel("BER for each Eb / N0 (dB)")
plt.title("bit Vs BER for each Eb / N0 (dB)")

plt.legend()
#plt.grid()
#plt.axis([0,10,1e-6,1])


plt.show()