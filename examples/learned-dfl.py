#!/usr/bin/env python

from dfl.dfl import *
from dfl.dynamic_system import *
from dfl.mpc import *
# from dfl.Transpose import Transpose

import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal

m = 1.0
k11 = 0.2
k13 = 2.0
b1  = 3.0

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        
        self.n_x = 2
        self.n_eta = 2
        self.n_u = 1

        self.n = self.n_x + self.n_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[0.0, 1.0],
                              [0.0, 0.0]])

        self.A_cont_eta = np.array([[0.0 , 0.0],
                                    [-1/m,-1/m]])

        self.B_cont_x = np.array([[0.0],[1.0]])

        # Limits for inputs and states
        self.x_min = np.array([-2.0,-2.0])
        self.x_max = np.array([2.0 ,2.0])

        self.u_min = np.array([-2.5])
        self.u_max = np.array([ 2.5])


    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c1(q):
        e = k11*q + k13*q**3
        return e

    @staticmethod
    def phi_r1(f):
        e = b1*np.sign(f)*f**2
        return e
    
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
        q,v = x[0], x[1]
        y = np.array([q,v])
        return y 
    
    @staticmethod
    def gkoop1(t,x,u):
        q,v = x[0], x[1]
        y = np.array([q,v,Plant1.phi_c1(q), Plant1.phi_r1(v)])
        return y  
    
    @staticmethod
    def gkoop2(t,x,u):
        q,v = x[0],x[1]

        y = np.array([q,v,q**2,q**3,q**4,q**5,q**6,q**7,
                      v**2,v**3,v**4,v**5,v**6,v**7,v**9,v**11,v**13,v**15,v**17,v**19,
                      v*q,v*q**2,v*q**3,v*q**4,v*q**5,
                      v**2*q,v**2*q**2,v**2*q**3,v**2*q**4,v**2
                      *q**5,
                      v**3*q,v**3*q**2,v**3*q**3,v**3*q**4,v**3*q**5])
        return y 


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
    return np.sin(3*t)

def square_u_func(y,t):
    return 0.5*signal.square(3 * t)

if __name__== "__main__":
    ################# DFL MODEL TEST ##############################################
    plant1 = Plant1()
    dfl1 = DFL(plant1, dt_data = 0.05, dt_control = 0.2)
    setattr(plant1, "g", Plant1.gkoop2)
    driving_fun = square_u_func
    T = 11.0

    dfl1.generate_data_from_random_trajectories( t_range_data = 5.0, n_traj_data = 100 )
    dfl1.learn_eta_fn()
    # dfl1.regress_New_Linear_Model()
    dfl1.generate_DFL_disc_model()
    dfl1.regress_K_matrix()

    # A_lrn = dfl1.model.A.weight.detach().clone().numpy()
    # H_lrn = dfl1.model.H.weight.detach().clone().numpy()
    # dfl1.lstsqAH()
    # A_lse = dfl1.model.A.weight.detach().numpy()
    # H_lse = dfl1.model.H.weight.detach().numpy()
    # print('A_lrn',A_lrn,'\nA_lse',A_lse,'\nH_lrn',H_lrn,'\nH_lse',H_lse)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.matshow((A_lrn-A_lse)/A_lse)
    # plt.show()
    # breakpoint()

    x_0 = np.array([0,0])

    t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, driving_fun, T)
    
    t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, driving_fun, T, continuous = False)
    t, u_koop, x_koop, y_koop = dfl1.simulate_system_koop(x_0, driving_fun, T)
    t, u_lrn, x_lrn, y_lrn = dfl1.simulate_system_learned(x_0, driving_fun, T)

    sse_dfl = np.sum(np.abs(y_nonlin[:,0]-y_dfl[:,0]))
    sse_kop = np.sum(np.abs(y_nonlin[:,0]-y_koop[:,0]))
    sse_lrn = np.sum(np.abs(y_nonlin[:,0]-y_lrn[:,0]))
    print(sse_dfl)
    print(sse_kop)
    print(sse_lrn)
    
    matplotlib.rcParams.update({'font.size': 22})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(t, y_nonlin[:,0], 'k', label='True')
    axs[0].plot(t, y_koop[:,0] ,'g-.', label='Koopman')
    axs[0].plot(t, y_dfl[:,0] ,'r-.', label='DFL')
    axs[0].plot(t, y_lrn[:,0] ,'b-.', label='L3')
    axs[0].legend(ncol=4,bbox_to_anchor=(0, 1, 1, 0))
    axs[0].tick_params(labelbottom = False, bottom = False)

    axs[1].plot(t, y_nonlin[:,1],'k')
    axs[1].plot(t, y_koop[:,1] ,'g-.')
    axs[1].plot(t, y_dfl[:,1],'r-.')
    axs[1].plot(t, y_lrn[:,1],'b-.')
    axs[1].tick_params(labelbottom = False, bottom = False)
  
    axs[2].plot(t, u_nonlin,'k')

    axs[2].set_xlabel('time')
    
    axs[0].set_ylabel('x')
    axs[1].set_ylabel('v')
    axs[2].set_ylabel('u')

    plt.show()

    # More graphs!
    # x = np.linspace(-0.2, 0.2)
    # v = np.linspace(-0.2, 0.2)
    # xx, vv = np.meshgrid(x,v, indexing='ij')
    
    # Fs = np.zeros(xx.shape)
    # Fd = np.zeros(xx.shape)
    # e1 = np.zeros(xx.shape)
    # e2 = np.zeros(xx.shape)

    # for xi in range(len(x)):
    #     for vi in range(len(v)):
    #         Fs[xi,vi] = Plant1.phi_c1(xx[xi,vi])
    #         Fd[xi,vi] = Plant1.phi_r1(vv[xi,vi])
    #         e = dfl1.eta_fn(np.array([xx[xi,vi],vv[xi,vi]]), np.array([0]))
    #         e1[xi,vi] = e[0]
    #         e2[xi,vi] = e[1]

    # matplotlib.rcParams.update({'font.size': 22})

    # fig = plt.figure()
    # axs0 = fig.add_subplot(221, projection='3d')
    # axs1 = fig.add_subplot(222, projection='3d')
    # axs2 = fig.add_subplot(223, projection='3d')
    # axs3 = fig.add_subplot(224, projection='3d')
    # axs = [axs0,axs1,axs2,axs3]

    # axs0.plot_surface(xx,vv,Fs, cmap=cm.hsv, rstride=1, cstride=1, linewidth=0, antialiased=False)
    # axs1.plot_surface(xx,vv,Fd, cmap=cm.hsv, rstride=1, cstride=1, linewidth=0, antialiased=False)
    # axs2.plot_surface(xx,vv,e1, cmap=cm.hsv, rstride=1, cstride=1, linewidth=0, antialiased=False)
    # axs3.plot_surface(xx,vv,e2, cmap=cm.hsv, rstride=1, cstride=1, linewidth=0, antialiased=False)

    # # axs0.title.set_text('(a)')
    # # axs1.title.set_text('(b)')
    # # axs2.title.set_text('(c)')
    # # axs3.title.set_text('(d)')

    # plt.tight_layout()
    # for ax in axs:
    #     ax.view_init(azim=0, elev=90)
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('v')
    #     ax.set_zticks([])
    #     ax.tick_params(left=False,
    #                 bottom=False,
    #                 labelleft=False,
    #                 labelbottom=False)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()