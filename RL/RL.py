import numpy as np
import random as rd
import os


dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ENV.map")
n = 15
refmap = np.genfromtxt(dataset_path, delimiter=',')
q_table = np.zeros((n,n,4))
r1 = 100
r2 = 30
c1 = -0.01
c2 = -20
c3 = -100
wall = -0.1
gamma = 0.9
alpha = 0.1
rew_ref = {'0':c1, '2':c2, '3':r1, '4':r2, '5':c3, '6':wall}
action_ref = {'U':[0,1], 'D':[0,-1],'L':[-1,0], 'R':[1,0]}
action_iref = {0:'U', 1:'D', 2:'L', 3:'R'}
epochs = 20



for i in range(epochs):
    state = [0,0]
    cost = 0
    while(True):
        new_state = state.copy()
        soft = np.exp(q_table[state[0]][state[1]])
        action_i = np.argmax(soft)
        action = action_iref[action_i]
        whereto = action_ref[action]
        new_state[0] += whereto[0]
        new_state[1] = whereto[1]
        if(new_state[0]<0 or new_state[0]>n or new_state[1]<0 or new_state[0]>15):
            q_table[state[1]][state[0]][action_i] = -100
        else:
            q_table[state[1]][state[0]][action_i] += alpha*(rew_ref[refmap[new_state[1]][new_state[0]]] + gamma * max())
                
        








