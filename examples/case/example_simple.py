#!/usr/bin/env python

import dfl.dynamic_system
import dfl.dynamic_model as dm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

m = 1.0
k11 = 0.2
k13 = 2.0
b1  = 3.0

class Plant1(dfl.dynamic_system.DFLDynamicPlant):
    
    def __init__(self):
        
        self.n_x = 2
        self.n_eta = 2
        self.n_u = 1

        self.n = self.n_x + self.n_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[0.0, 1.0],
                              [0.0, 0.0]])

        self.A_cont_eta = np.array([[0.0, 0.0],
                               [-1/m,-1/m]])

        self.B_cont_x = np.array([[0.0],[1.0]])

        # Limits for inputs and states
        self.x_min = np.array([-2.0,-2.0])
        self.x_max = np.array([2.0 ,2.0])

        self.u_min = np.array([-2.5])
        self.u_max = np.array([ 2.5])

        # Hybrid model
        self.P =  np.array([[1, 1]])

        self.A_cont_eta_hybrid =   self.A_cont_eta.dot(np.linalg.pinv(self.P))

    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c1(q):
        e = k11*q + k13*q**3
        return e

    @staticmethod
    def phi_r1(f):
        # e = b1*np.sign(f)*np.abs(f)*np.abs(f)
        e = b1*np.sign(f)*f**2
        return e

    @staticmethod
    def phi_rc(q,v):
        return 5*v*np.abs(q)
    
    # nonlinear state equations
    def f(self,t,x,u):
        x_dot = np.zeros(x.shape)
        q,v = x[0],x[1]
        x_dot[0] = v
        x_dot[1] = -self.phi_r1(v) -self.phi_c1(q) + u 

        return x_dot

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        return dm.Koopman.gkoop1(x)
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q,v = x[0],x[1]
        
        eta = np.zeros(self.n_eta)
        eta[0] = self.phi_c1(q)
        eta[1] = self.phi_r1(v)

        return eta

###########################################################################################

#Dummy forcing laws
def zero_u_func(y,t):
    return 1 

def rand_u_func(y,t):
    return np.random.normal(0.0,0.3)

def sin_u_func(y,t):
    return 0.5*signal.square(3 * t)
    # return np.sin(3*t) 

if __name__== "__main__":
    driving_fun = sin_u_func
    plant1 = Plant1()
    x_0 = np.zeros(plant1.n_x)
    fig, axs = plt.subplots(2, 1)

    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()
    t, u, x_tru, y_tru = tru.simulate_system(x_0, driving_fun, 10.0)
    axs[0].plot(t, x_tru[:,0], 'k-', label='Ground Truth')

    koo = dm.Koopman(plant1, observable='filippos')
    koo.learn(data)
    _, _, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, 10.0)
    axs[0].plot(t, x_koo[:,0], 'g-.', label='Koopman')

    dfl = dm.DFL(plant1)
    dfl.learn(data)
    _, _, x_dfl, y_dfl = dfl.simulate_system(x_0, driving_fun, 10.0)
    axs[0].plot(t, x_dfl[:,0], 'r-.', label='DFL')

    lrn = dm.L3(plant1, 2, ac_filter=False)
    lrn.learn(data)
    _, _, x_lrn, y_lrn = lrn.simulate_system(x_0, driving_fun, 10.0)
    axs[0].plot(t, x_lrn[:,0], 'b-.', label='L3')

    axs[0].legend()
  
    axs[1].plot(t, u, 'k')

    axs[1].set_xlabel('time')
    
    axs[0].set_ylabel('q')
    axs[1].set_ylabel('u')

    plt.show()