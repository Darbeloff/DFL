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
        self.x_min = np.array([-2.0,-2.0])
        self.x_max = np.array([2.0 ,2.0])

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
        return np.copy(x)
    
    # @staticmethod
    # def gkoop1(t,x,u):
    #     q,v = x[0], x[1]
    #     y = np.array([q,v,Plant1.phi_c1(q), Plant1.phi_r1(v)])
    #     return y  
    
    # @staticmethod
    # def gkoop2(t,x,u):
    #     q,v = x[0],x[1]

    #     y = np.array([q,v,q**2,q**3,q**4,q**5,q**6,q**7,
    #                   v**2,v**3,v**4,v**5,v**6,v**7,v**9,v**11,v**13,v**15,v**17,v**19,
    #                   v*q,v*q**2,v*q**3,v*q**4,v*q**5,
    #                   v**2*q,v**2*q**2,v**2*q**3,v**2*q**4,v**2
    #                   *q**5,
    #                   v**3*q,v**3*q**2,v**3*q**3,v**3*q**4,v**3*q**5])
    #     return y 


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

    # def phi_hybrid(self,t,x,u):
    #     '''
    #     outputs the values of the auxiliary variables
    #     '''
    #     q,v = x[0],x[1]
    #     eta = np.zeros(self.n_eta)
    #     eta[0] = self.phi_c1(q) + self.phi_r1(v)

    #     return eta

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
    # plant1 = Plant1()

    # dfl1 = DFL(plant1)
    # setattr(plant1, "g", Plant1.gkoop1)

    # # ########## KOOPMAN MPC TEST
    # dfl1.generate_data_from_random_trajectories()
    # # dfl1.generate_H_matrix()
    # # dfl1.generate_N4SID_model()
    # dfl1.generate_K_matrix()

    # x_0 = np.array([1,0])
    # T = 10.0

    # # Objective function
    # Q = sparse.diags([1., 0., 0., 0.])
    # QN = Q
    # R = 0.01*sparse.eye(1)

    # # Initial and reference states
    # u_minus = np.zeros((dfl1.plant.N_u,1))
    # x0 = dfl1.plant.g(0.0, x_0, u_minus)
    # xr = np.array([0.5 ,0., 0., 0.])

    # x_min = np.array([-3.,-3.,-3.,-3.])
    # x_max = np.array([3.,3.,3.,3.])

    # # dummy reference trajectory
    # t_traj = np.arange(0,T,0.05)
    # x1_traj = np.sin(0.5*t_traj)
    # x_traj = np.vstack((x1_traj,0*x1_traj,0*x1_traj,0*x1_traj)).T

    # # mpc = MPC(dfl1.A_koop,
    # #           dfl1.B_koop,
    # #           x_min, x_max,
    # #           plant1.u_min,
    # #           plant1.u_max)

    # mpc = MPC(dfl1.A_koop,
    #           dfl1.B_koop,
    #           x_min, x_max,
    #           plant1.u_min,
    #           plant1.u_max,
    #           N = 20)
    
    # mpc.setup_new_problem(Q, QN, R, t_traj, x_traj, x0)

    # x_0 = np.array([1,0])
    # t, u, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0,  mpc.control_function, T)
    # # t, u, x_koop1, y_koop = dfl1.simulate_system_koop(x_0, mpc.control_function, 10.0)
    

    # # mpc.control_function
    # fig, axs = plt.subplots(3, 1)

    # axs[0].plot(t, x_nonlin[:,0], 'b')
    # axs[0].plot(t_traj, x_traj[:,0], 'b--')

    # axs[1].plot(t, x_nonlin[:,1], 'r')
    # axs[2].plot(t, u[:,0], 'g')

    # axs[2].set_xlabel('time')
    
    # axs[0].set_ylabel('x')
    # axs[1].set_ylabel('v')
    # axs[2].set_ylabel('u')
    # plt.show()

    # exit()

    ################# DFL MODEL TEST ##############################################
    plant1 = Plant1()
    dfl1 = DFL(plant1, dt_data = 0.05, dt_control = 0.2)
    setattr(plant1, "g", Plant1.g)

    dfl1.generate_data_from_random_trajectories( t_range_data = 5.0, n_traj_data = 100 )
    dfl1.generate_DFL_disc_model()
    dfl1.regress_K_matrix()

    # x_0 = np.random.uniform(plant1.x_init_min,plant1.x_init_max)
    x_0 = np.array([0])
    seed = np.random.randint(5)

    np.random.seed(seed = seed)
    t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, sin_u_func, 10.0)
    
    np.random.seed(seed = seed)
    t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, sin_u_func, 10.0,continuous = False)
    # t, u_koop, x_koop, y_koop = dfl1.simulate_system_koop(x_0, sin_u_func, 10.0)
    
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(t, y_nonlin[:,0], 'b')
    axs[0].plot(t, y_dfl[:,0] ,'b-.')
    # axs[0].plot(t, y_koop[:,0] ,'b--')

    # axs[1].plot(t, y_nonlin[:,1],'r')
    # axs[1].plot(t, y_dfl[:,1],'r-.')
    # axs[1].plot(t, y_koop[:,1] ,'r--')
  
    axs[2].plot(t, u_nonlin,'g')
    axs[2].plot(t, u_dfl,'r-.')
    # axs[2].plot(t, u_koop,'b--')

    axs[2].set_xlabel('time')
    
    axs[0].set_ylabel('x')
    axs[1].set_ylabel('v')
    axs[2].set_ylabel('u')

    plt.show()