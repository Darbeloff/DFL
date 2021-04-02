#!/usr/bin/env python

from dfl.dfl import *
from dfl.dynamic_system import *
from dfl.mpc import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

m = 1.0
k11 = 0.2
k13 = 2.0
b1  = 3.0

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        
        self.n_x = 1
        self.n_eta = 2
        self.n_u = 1

        self.n = self.n_x + self.n_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[0.0]])

        self.A_cont_eta = np.array([[0.0, 1.0]])

        self.B_cont_x = np.array([[0.0]])

        # Limits for inputs and states
        self.x_min = np.array([-2.0])
        self.x_max = np.array([ 2.0])

        self.u_min = np.array([-2.5])
        self.u_max = np.array([ 2.5])

        # Hybrid model
        self.P =  np.array([[1, 1]])

        self.A_cont_eta_hybrid =   self.A_cont_eta.dot(np.linalg.pinv(self.P))


    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c(q):
        return np.sign(q)*q**2

    @staticmethod
    def phi_r(f):
        return 2*(1/(1+np.exp(-4*f))-0.5)
    
    # nonlinear state equations
    def f(self,t,x,u):
        x_dot = np.zeros(x.shape)
        eta = self.phi(t,x,u)
        x_dot[0] = eta[1]

        return x_dot

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        if not isinstance(u,np.ndarray):
            u = np.array([u])
            
        q = x[0]
        ec = Plant1.phi_c(q)
        er = u[0]-ec
        f = Plant1.phi_r(er)

        # return np.array([q,ec,f])

        return np.copy(x)

    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        if not isinstance(u,np.ndarray):
            u = np.array([u])
            
        q = x[0]
        ec = Plant1.phi_c(q)
        er = u[0]-ec
        f = Plant1.phi_r(er)
        
        eta = np.zeros(self.n_eta)
        eta[0] = ec
        eta[1] = f

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
    ################# DFL MODEL TEST ##############################################
    plant1 = Plant1()
    dfl1 = DFL(plant1, dt_data = 0.05, dt_control = 0.2, n_koop=1)
    # setattr(plant1, "g", Plant1.g)
    driving_fun = sin_u_func

    dfl1.generate_data_from_random_trajectories( t_range_data = 5.0, n_traj_data = 110 )
    dfl1.clean_anticausal_eta()
    dfl1.generate_DFL_disc_model()
    dfl1.regress_K_matrix()

    # x_0 = np.random.uniform(plant1.x_init_min,plant1.x_init_max)
    x_0 = np.array([0])
    seed = np.random.randint(5)

    np.random.seed(seed = seed)
    t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, driving_fun, 10.0)
    
    np.random.seed(seed = seed)
    t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, driving_fun, 10.0,continuous = False)
    t, u_koop, x_koop, y_koop = dfl1.simulate_system_koop(x_0, driving_fun, 10.0)
    
    fig, axs = plt.subplots(2, 1)

    # axs[0].plot(t, x_nonlin[:,0], 'k', label='True')
    axs[0].plot(t, x_dfl[:,0] ,'g-.', label='DFL')
    # axs[0].plot(t, x_koop[:,0] ,'b--', label='Koopman')
    axs[0].legend()

    # axs[1].plot(t, y_nonlin[:,1],'r')
    # axs[1].plot(t, y_dfl[:,1],'r-.')
    # axs[1].plot(t, y_koop[:,1] ,'r--')
  
    axs[1].plot(t, u_nonlin,'k')
    # axs[2].plot(t, u_dfl,'r-.')
    # axs[2].plot(t, u_koop,'b--')

    axs[1].set_xlabel('time')
    
    axs[0].set_ylabel('q')
    axs[1].set_ylabel('u')

    plt.show()