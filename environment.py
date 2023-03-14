import os
import sys
import numpy as np
from scipy.signal import cont2discrete, lti, dlti, dstep
import scipy.optimize
import gym
import random
from copy import copy, deepcopy
from collections import deque
from functools import partial
from matplotlib import pyplot as plt
import copy

def deadband_func(omega, LL, UL):
    out = deepcopy(omega)
    out[(out >= LL) & (out <= UL)] = 0.0
    out[(out >= UL)] -= UL
    out[(out <= LL)] -= LL
    return out

def power_flow(data, x):
    A_test = data[0]
    B_test = data[1]
    net_load = data[2]
    return A_test @ np.concatenate((np.array([0.0]), x[1:])) \
            + B_test @ np.concatenate((np.array([x[0]]), net_load[1:]))

def step_dynamics(sys, costs_params, net_load, s, u, rocof_old):
    n = int(len(s) / 2)
    A = sys[0]
    B = sys[1]
    dt = sys[4]
    cost_theta = costs_params[0]
    cost_omega = costs_params[1]
    cost_u = costs_params[2]
    cost_rocof = costs_params[3]
    omega_old = s[n:]
    theta = s[0:n]
    omega = s[n:]
    costs = cost_theta * np.linalg.norm(theta) + cost_omega * np.linalg.norm(deadband_func(omega, -0.01, 0.01), np.inf) \
        + cost_u * np.linalg.norm(u) + cost_rocof * np.linalg.norm(deadband_func(rocof_old, -0.02, 0.02), np.inf) #give more to u
    
    s1 = A @ s + B @ (u + net_load)
    rocof_new = (s1[n:] - omega_old) / dt #rocof means rate of change of frequency
    # theta = s1[0:n]
    # omega = s1[n:]
    # costs = cost_theta * np.linalg.norm(theta) + cost_omega * np.linalg.norm(deadband_func(omega, -0.01, 0.01), np.inf) \
    # + cost_u * np.linalg.norm(u)+ cost_rocof * np.linalg.norm(deadband_func(rocof_new, -0.02, 0.02), np.inf)
    return s1, s1[0:n], s1[n:], rocof_new, -costs

class Hybrid_system(gym.Env):
    def __init__(self):
        T = 200  #200 seconds of forecast, each time step has 4 seconds of duration
        self.N = 11  #Number of nodes
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(2*self.N,), dtype=np.float32)
        self.action_dim = self.N
        self.action_space = gym.spaces.Box(-5,5,shape=(self.action_dim,), dtype=np.float32)
        droop = 0.5#40.0
        self.D = droop*np.eye(self.N)# % 1.5%/% droop control.
        v_base = 230e3
        s_base = 100e6
        s_base_trans = 900e6
        Z_base = v_base**2/(s_base)  #230kV^2/ (100 MVA)
        Z_base_trans = v_base**2/(s_base_trans)
        imp_trans_pu =  0.15j
        imp_trans_nopu = imp_trans_pu * Z_base_trans  #Ohm/km
        self.suscept_trans = np.imag(1/(0.15j))  #1/p.u.
        line_imp_pu = 0.0001 + 0.001j
        line_imp_nopu = line_imp_pu * Z_base  #Impedance not in pu
        self.suscept_line = np.imag(1/(line_imp_pu))  # km/p.u.
        self.LT()
        self.Inertia_matrix()
        p = 2*700 + 611 + 1050 + 350 + 719  
        q = 208 + 293 + 164 +284 + 69 + 133 
        p_gen = np.array([611, 600, 0, 0, 719, 350, 0, 0, 700, 700, 0,])/100  #in pu
        p_load = np.array([0, 0, 400, 567, 0,  0, 490, 1000, 0, 0, 1570, ])/100  #in pu
        self.net_load = (p_gen - p_load)
        p_avg = p/self.N 
        q_avg = q/self.N 
        p_gens = p_avg*np.ones(self.N) 
        q_gens = q_avg*np.ones(self.N) 
        s = np.sqrt(p_gens ** 2 + q_gens ** 2) 
        s_matrix = np.diag(s)  # basically power rating assumed at each node. This one can be edited to simulate different systems (i.e. different dynamical systems!)
        self.f_base = 50 
        self.w_base = 2*np.pi*self.f_base 
        self.M = np.zeros((self.N, self.N, 10)) 
        for i in range(10):
            self.M[:,:,i] = self.H[:,:,i] 
        self.system_init()
        self.costs_params = np.asarray([0.0, 1.0e3, 10, 0.00])

    def LT(self):
        self.L = np.zeros((self.N+1,self.N+1))
        L1_1 = 1/25 * self.suscept_line + self.suscept_trans 
        self.L[1,1] = L1_1 
        L1_3 = - 1/25 * self.suscept_line - self.suscept_trans 
        self.L[1,3] = L1_3 

        L2_3 = - self.suscept_trans 
        self.L[2,3] = L2_3 

        self.L[2,2] = -L2_3 

        L3_1 = L1_3 
        self.L[3,1] = L3_1 
        L3_2 = L2_3 
        self.L[3,2] = L3_2 
        L3_3 = -L1_3 -L2_3 + 1/10*self.suscept_line 
        self.L[3,3] = L3_3 
        L3_4 = -1/10*self.suscept_line 
        self.L[3,4] = L3_4 

        L4_3 = L3_4 
        self.L[4,3] = L4_3 
        L4_4 = 1/10*self.suscept_line + 1/110*self.suscept_line + 1/110*self.suscept_line  #%+ suscept_shunt8 
        self.L[4,4] = L4_4 
        L4_8 = -1/110*self.suscept_line 
        self.L[4,8] = L4_8 
        L4_11 = -1/110*self.suscept_line 
        self.L[4,11] = L4_11 

        L5_5 = self.suscept_trans + 1/25*self.suscept_line 
        self.L[5,5] = L5_5 
        L5_7 = -1/25*self.suscept_line - self.suscept_trans 
        self.L[5,7] = L5_7 

        L6_6 = self.suscept_trans 
        self.L[6,6] = L6_6 
        L6_7 = - self.suscept_trans 
        self.L[6,7] = L6_7 

        self.L[7,5] = L5_7 
        L7_6 = - self.suscept_trans 
        self.L[7,6] = L7_6 
        L7_7 = 1/25*self.suscept_line + 2*self.suscept_trans + 1/10*self.suscept_line 
        self.L[7,7] = L7_7 
        L7_8 = -1/10*self.suscept_line 
        self.L[7,8] = L7_8 

        L8_4 = L4_8 
        self.L[8,4] = L8_4 
        L8_7 = L7_8 
        self.L[8,7] = L8_7 
        L8_8 = 1/110*self.suscept_line + 1/110*self.suscept_line + 1/10*self.suscept_line #  %+ 1/110*suscept_line  % + suscept_shunt8 
        self.L[8,8] = L8_8 
        L8_11 = L4_8 
        self.L[8,11] = L8_11 

        L9_9 = self.suscept_trans + 1/25*self.suscept_line 
        self.L[9,9] = L9_9 
        L9_11 = -self.suscept_trans -  1/25*self.suscept_line 
        self.L[9,11] = L9_11 

        L10_11 = -self.suscept_trans 
        self.L[10,11] = L10_11 
        L10_10 = self.suscept_trans 
        self.L[10,10] = L10_10 

        self.L[11,4] = L4_11 
        self.L[11,8] = L8_11 
        L11_9 = L9_11 
        self.L[11,9] = L11_9 
        L11_10 = L10_11 
        self.L[11,10] = L11_10 
        L11_11 = 2*self.suscept_trans + 1/25*self.suscept_line + 2*1/110*self.suscept_line 
        self.L[11,11] = L11_11 

        #Change to make Glover system of equations -B*delta = P, that will be for
        #us L*delta = P. Page 353 Glover Textbook.
        #Python starts from zero, not one.
        self.L = -self.L[1:12,1:12]
    
    def Inertia_matrix(self):
        self.H = np.zeros((self.N, self.N, 10))  # 10 hybrid modes
        self.H[:,:,0] = 0.1*np.eye(self.N, self.N)   # 0.1 sec of inertia. Extreme ~ 1% of thermal units with 10 secs of inertia, the rest renewables with 0 s of inertia
        self.H[:,:,1] = 0.5*np.eye(self.N, self.N) 
        self.H[:,:,2] = 1*np.eye(self.N, self.N) 
        self.H[:,:,3] = 1.5*np.eye(self.N, self.N) 
        self.H[:,:,4] = 2*np.eye(self.N, self.N) 
        self.H[:,:,5] = 2.5*np.eye(self.N, self.N) 
        self.H[:,:,6] = 3*np.eye(self.N, self.N) 
        self.H[:,:,7] = 3.5*np.eye(self.N, self.N) 
        self.H[:,:,8] = 5*np.eye(self.N, self.N) 
        self.H[:,:,9] = 9*np.eye(self.N, self.N) 

    def system_init(self):
        self.d_system_list = [] 
        self.A_list = []
        self.B_list = []
        #I think there might be some bug, the A_test, B_test is changing all the time
        for i in range(10):
            inertia_mode = i
            A_test = np.zeros((2*self.N, 2*self.N)) 
            B_test = np.zeros((2*self.N, self.N)) 
            C_test = np.zeros((1, 2*self.N)) 
            D_test = np.zeros((1,self.N))
            M_test = self.M[:,:,inertia_mode] 

            A_test[0:self.N, self.N:2*self.N] = self.w_base*np.eye(self.N) 
            A_test[self.N:2*self.N, 0:self.N] =  -np.linalg.inv(M_test) @ self.L 
            A_test[self.N:2*self.N, self.N:2*self.N] = -np.linalg.inv(M_test) @ self.D 

            B_test[self.N:2*self.N, :] = np.linalg.inv(M_test) 
            dt = 0.001
            d_system = cont2discrete((A_test, B_test, C_test, D_test), dt)
            self.A_list.append(A_test)
            self.B_list.append(B_test)
            self.d_system_list.append(d_system)

    def step(self, action): 
        done = False 
        next_state, angle, omega, rocof, reward = step_dynamics(self.d_system, self.costs_params, self.net_load_new, self.state, action, self.rocof_old)
        self.state = deepcopy(next_state)
        self.state = np.asarray(self.state)
        if np.max(np.abs(self.state))<0.005:
            done = True
            print('success')
        self.rocof_old = deepcopy(rocof)
        info = {'A':self.A_list[self.sys_id], 'B':self.B_list[self.sys_id], 'dis_A':self.d_system_list[self.sys_id][0], 'dis_B':self.d_system_list[self.sys_id][1]}
        return next_state, reward, done, info
    def seed(self, seed):
        np.random.seed(seed)
    
    def reset(self): #sample different initial volateg conditions during training
        #solve power flow
        x0 = np.zeros(2*self.N) 
        sys_ix = np.random.randint(0, 10)
        self.d_system = self.d_system_list[sys_ix]
        # print(sys_ix)
        data = [self.A_list[sys_ix], self.B_list[sys_ix], self.net_load]
        pf_partial = partial(power_flow, data)
        x_sol = scipy.optimize.broyden1(pf_partial, x0, f_tol=1e-10)
        # Update the generation of slack bus
        self.net_load_new = deepcopy(self.net_load)
        self.net_load_new[0] = x_sol[0]
        state = deepcopy(x_sol)
        state[0] = 0.0
        # self.net_load_new += 0.1*np.random.randn(len(self.net_load))
        self.freq_ix = np.random.randint(self.N, 2*self.N)
        # state[self.freq_ix] = 2*(np.random.rand() - 0.5)/(self.f_base)
        self.state = deepcopy(state)

        self.state = np.asarray(self.state)
        self.state += np.random.randn(self.state.shape[0])
        self.rocof_old = np.zeros(self.N)
        info = {'A':self.A_list[sys_ix], 'B':self.B_list[sys_ix], 'dis_A':self.d_system_list[sys_ix][0], 'dis_B':self.d_system_list[sys_ix][1]}
        self.sys_id = sys_ix
        a = np.zeros(11)
        for i in range(20):
            self.state,_,_,_ = self.step(a)
        #select system
        # return self.state, self.d_system_list[sys_ix][0], self.d_system_list[sys_ix][1]
        return self.state

if __name__ == "__main__":
    hyb_sys = Hybrid_system()
    for j in range(4):
        hyb_sys.seed(j)
        s = hyb_sys.reset()
        
        a = np.zeros(11)
        states_trajectory2 = []
        states_trajectory2.append(s[11:])
        T = 100
        N=11
        for i in range(T):
            a = np.zeros(11)
            s2, reward,_,_ = hyb_sys.step(a)
            s = deepcopy(s2)
            states_trajectory2.append(s[11:])
        plt.plot(states_trajectory2)
        plt.show()