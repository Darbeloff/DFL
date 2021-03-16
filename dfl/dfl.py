#!/usr/bin/env python
import sys

from abc import ABC, abstractmethod 
import numpy as np 
from numpy.linalg import lstsq
from scipy.linalg import expm
from numpy.linalg import inv

from scipy.integrate import ode
from scipy.signal import savgol_filter
from scipy.signal import cont2discrete
from scipy.linalg import eigvals
import copy

import matplotlib.pyplot as plt

np.set_printoptions(precision = 4)
np.set_printoptions(suppress = True)

class DFL():
    
    def __init__(self, dynamic_plant, dt_data = 0.05, dt_control = 0.1):
        
        self.plant = dynamic_plant
        self.dt_data = dt_data 
        self.dt_control = dt_control 

    def regress_H_cont_matrix(self):
        '''
        regress the H matrix for DFL model
        '''

        Xi = np.concatenate((self.x_data.reshape(-1, self.x_data.shape[-1]),
                             self.eta_data.reshape(-1, self.eta_data.shape[-1]),
                             self.u_data.reshape(-1, self.u_data.shape[-1])),axis=1)
        
        Eta_dot = self.eta_dot_data.reshape(-1, self.eta_dot_data.shape[-1])

        self.H_cont = lstsq(np.array(Xi,dtype='float'),np.array(Eta_dot,dtype='float'),rcond=None)[0].T
        self.H_cont_x   = self.H_cont[:,:self.plant.n_x]
        self.H_cont_eta = self.H_cont[:,self.plant.n_x:self.plant.n_x+self.plant.n_eta]
        self.H_cont_u   = self.H_cont[:,self.plant.n_x+self.plant.n_eta:self.plant.n_x+self.plant.n_eta+self.plant.n_u]

        self.convert_DFL_continuous_to_discrete()

    def generate_DFL_disc_model(self,method = 'LS'):
        '''
        regress the H matrix for DFL model
        '''
        omega = np.concatenate((self.X_minus.reshape(-1, self.X_minus.shape[-1]),
                                self.Eta_minus.reshape(-1, self.Eta_minus.shape[-1]),
                                self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T
        
        Y = self.Eta_plus.reshape(-1, self.Eta_plus.shape[-1]).T

        if method == 'LS':
            H_disc = lstsq(omega.T,Y.T,rcond=None)[0].T
            
            H_disc_x   = H_disc[:,:self.plant.n_x]
            H_disc_eta = H_disc[:,self.plant.n_x:self.plant.n_x+self.plant.n_eta]
            H_disc_u   = H_disc[:,self.plant.n_x+self.plant.n_eta:self.plant.n_x+self.plant.n_eta+self.plant.n_u]

            (A_disc_x, B_disc_x,_,_,_) = cont2discrete((self.plant.A_cont_x, self.plant.B_cont_x, 
                                                np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                self.dt_data)
            # check: changed n_u to n_eta
            (_,A_disc_eta ,_,_,_)   = cont2discrete((self.plant.A_cont_x, self.plant.A_cont_eta, 
                                          np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                self.dt_data)

            # F = np.block([[self.plant.A_cont_x,self.plant.A_cont_eta,self.plant.B_cont_x],
            #               [np.zeros((self.plant.N_eta+self.plant.N_u,self.plant.N_eta+self.plant.N_u+self.plant.N_eta))]])
            # print(expm(F*self.dt_data))

            self.A_disc_dfl =  np.block([[A_disc_x  , A_disc_eta],
                                                 [H_disc_x  , H_disc_eta]])

            self.B_disc_dfl = np.block([[B_disc_x],
                                    [H_disc_u]])

    def convert_DFL_continuous_to_discrete(self):

        A_cont_full = np.block([[self.plant.A_cont_x, self.plant.A_cont_eta],
                                [self.H_cont_x,   self.H_cont_eta]])

        B_cont_full = np.block([[self.plant.B_cont_x],
                                [self.H_cont_u]])

        # A_disc = expm(A_full*self.dt_data)
        # B_disc = inv(A_full).dot(expm(A_full*self.dt_data) - np.eye(A_full.shape[0])).dot(B_full)

        (A_disc,B_disc,_,_,_) = cont2discrete((A_cont_full, B_cont_full, 
                                                 np.zeros(self.plant.N_x), np.zeros(self.plant.N_u)),
                                                 self.dt_data)

        self.A_disc_dfl = A_disc
        self.B_disc_dfl = B_disc

    def regress_K_matrix(self):

        omega = np.concatenate((self.Y_minus.reshape(-1, self.Y_minus.shape[-1]),
                                self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T
        
        Y = self.Y_plus.reshape(-1, self.Y_plus.shape[-1]).T

        G = lstsq(omega.T,Y.T,rcond=None)[0].T
        
        self.A_disc_koop = G[: , :self.Y_plus.shape[-1]] 
        self.B_disc_koop = G[: , self.Y_plus.shape[-1]:]

    def copy_into_minus_plus(data):
        minus = np.copy(data[:, :-1,:])
        plus  = np.copy(data[:,1:  ,:])
        return minus, plus

    def clean_anticausal_eta(self):
        u   = self.  u_data.reshape(-1,self.  u_data.shape[-1]).T
        eta = self.eta_data.reshape(-1,self.eta_data.shape[-1]).T
        self.D = np.linalg.lstsq(np.matmul(u,u.T),np.matmul(u,eta.T),rcond=None)[0].T

        eta_data_shape = self.eta_data.shape
        self.eta_data = self.eta_data.reshape(-1, self.eta_data.shape[-1]).T
        self.eta_data = self.eta_data-np.matmul(self.D, u)
        self.eta_data = self.eta_data.T.reshape(eta_data_shape)

        for i in range(eta_data_shape[0]):
            eta_array = self.eta_data[i]
            self.eta_dot_data[i] = savgol_filter(eta_array,
                                                 window_length = 5, polyorder = 3,
                                                 deriv = 1, axis=0)/self.dt_data

        self.Eta_minus, self.Eta_plus = DFL.copy_into_minus_plus(self.eta_dot_data)

    def generate_sid_model(self,xi_order):

        U = self.U_minus.reshape(-1, self.U_minus.shape[-1]).T
        Y = self.X_minus.reshape(-1, self.X_minus.shape[-1]).T
        
        if len(Y.shape) == 1:
            Y=Y.T
            Y = np.expand_dims(Y, axis=0)

        method = 'N4SID'
        sys_id = system_identification(Y, U, method, SS_D_required = True, SS_fixed_order = xi_order)

        self.A_disc_sid,self.B_disc_sid,self.C_disc_sid,self.D_disc_sid = sys_id.A, sys_id.B, sys_id.C, sys_id.D

    def generate_hybrid_model(self,xi_order):

        U = np.concatenate((self.X_minus.reshape(-1, self.X_minus.shape[-1]),
                            self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T

        Y_temp = self.Eta_minus.reshape(-1, self.Eta_minus.shape[-1]).T

        Y = self.plant.P.dot(Y_temp)
        
        if len(Y.shape) == 1:
            Y = Y.T
            Y = np.expand_dims(Y, axis=0)

        (A_disc_x, B_disc_x,_,_,_) = cont2discrete((self.plant.A_cont_x, self.plant.B_cont_x, 
                                                np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                self.dt_data)
            
        (_,A_disc_eta_hybrid,_,_,_)   = cont2discrete((self.plant.A_cont_x, self.plant.A_cont_eta_hybrid, 
                                          np.zeros(self.plant.n_x), np.zeros(self.plant.n_eta)),
                                                self.dt_data)

        method = 'N4SID'
        sys_id = system_identification(Y, U, method, SS_D_required = True, SS_fixed_order = int(xi_order)) #, IC='AICc')#

        # SS_fixed_order = self.plant.N_eta,
        A_til,B_til,C_til,D_til = sys_id.A, sys_id.B, sys_id.C, sys_id.D
        
        B_til_1 = B_til[:,:self.plant.n_x]
        B_til_2 = B_til[:,self.plant.n_x:]
        D_til_1 = D_til[:,:self.plant.n_x]
        D_til_2 = D_til[:,self.plant.n_x:]

        A1 = A_disc_x + A_disc_eta_hybrid.dot(D_til_1)
        A2 = A_disc_eta_hybrid.dot(C_til)
        B1 = B_disc_x+ A_disc_eta_hybrid.dot(D_til_2)

        self.A_disc_hybrid_full =  np.block([[A1     ,  A2],
                                             [B_til_1,  A_til ]])

        self.B_disc_hybrid_full = np.block([[B1],
                                       [B_til_2]])

        self.C_til = C_til
        self.D_til_1 = D_til_1
        self.D_til_2 = D_til_2

    def f_cont_dfl(self,t,xi,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        x   = xi[:self.plant.n_x]
        eta = xi[self.plant.n_x:self.plant.n_x + self.plant.n_eta]
        
        x_dot   = np.dot(self.plant.A_cont_x,x) +  np.dot(self.plant.A_cont_eta, eta) + np.dot(self.plant.B_cont_x,u)
        eta_dot = np.dot(self.H_cont_x,x) +  np.dot(self.H_cont_eta, eta) + np.dot(self.H_cont_u,u)

        return np.concatenate((x_dot,eta_dot))

    def f_disc_dfl(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        y_plus = np.dot(self.A_disc_dfl,x) + np.dot(self.B_disc_dfl, u)

        return y_plus

    def f_disc_koop(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        y_plus = np.dot(self.A_disc_koop,x) + np.dot(self.B_disc_koop, u)

        return y_plus

    def f_disc_hybrid(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        x_plus = np.dot(self.A_disc_hybrid_full,x) + np.dot(self.B_disc_hybrid_full, u)

        return x_plus
    
    def f_disc_sid(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        x_plus = np.dot(self.A_disc_sid, x) + np.dot(self.B_disc_sid, u)

        return x_plus
    
    def g_disc_hybrid(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        if len(u.shape) > 1:
            u = u.flatten()

        y = np.dot(self.C_disc_sid, x) + np.dot(self.D_disc_sid, u)
        
        return y
    
    def g_disc_sid(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        if len(u.shape) > 1:
            u = u.flatten()

        y = np.dot(self.C_disc_sid, x) + np.dot(self.D_disc_sid, u)
        
        return y
    
    def g_disc_hybrid(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        if len(u.shape) > 1:
            u = u.flatten()

        eta = np.dot(self.C_til, x[self.plant.N_x:]) + np.dot(self.D_til_1, x[:self.plant.N_x]) + np.dot(self.D_til_2, u)

        y = np.concatenate((x[:self.plant.N_x],eta))

        return y

    @staticmethod
    def simulate_system(x_0, u_minus, t_f, dt, u_func, dt_control, f_func, g_func, continuous = True):
        '''
        Simulate a system in continuous time
        Arguments:
        x_0: initial state
        u_func: control function
        t_f: final time

        '''
        # initial time and input
        t = 0.0

        # create numerical integration object
        if continuous:
            r = ode(f_func).set_integrator('vode', method = 'bdf', max_step = 0.001)
            r.set_initial_value(x_0,t)

        t_array = []
        x_array = []
        u_array = []
        y_array = []
        
        # initial state and 
        x_t = copy.copy(x_0)
        y_t = g_func(t, x_t, u_minus)

        u_t = u_func(y_t, t)

        t_array.append(t)
        x_array.append(x_t)
        u_array.append([u_t])
        y_array.append(g_func(t,x_t,u_minus))

        t_control_last = 0
        #Simulate the system
        while t < t_f:

            if continuous:
                r.set_f_params(u_t).set_jac_params(u_t)
                x_t = r.integrate(r.t + dt)
            else:
                x_t = f_func(t, x_t, u_t)

            t = t + dt
            y_t = g_func(t, x_t, u_t)
            
            if t - t_control_last > dt_control:
                t_control_last = copy.copy(t) 
                u_t = u_func(g_func(t, x_t, u_t), t)

            t_array.append(t)
            x_array.append(x_t)
            u_array.append([u_t])
            y_array.append(y_t)

        return np.array(t_array), np.array(u_array), np.array(x_array), np.array(y_array)

    def generate_data_from_random_trajectories(self, t_range_data = 10.0, n_traj_data = 50,
                                                     x_0 = None, plot_sample = False):
        '''
        create random data to train DFL and other dynamic system models
        '''
        t_data = []
        x_data = []
        u_data = []
        eta_data = []
        eta_dot_data = []

        
        X_minus_data = []
        U_minus_data = []
        Y_minus_data = []
        Eta_minus_data = []

        X_plus_data  = []
        U_plus_data  = []
        Y_plus_data  = []
        Eta_plus_data  = []

        for i in range(n_traj_data):
            
            # define initial conitions and range of time
            t_0 = 0.0
            t_f = t_range_data

            if x_0 is None:
                x_0 = np.random.uniform(self.plant.x_min,self.plant.x_max)

            #initialize the ode integrator
            r = ode(self.plant.f).set_integrator('dopri5')
            r.set_initial_value(x_0,t_0)

            t_array = []
            x_array = []
            u_array = []
            eta_array = []
            y_array = []

            t_control_last = -10000000
            u_t = np.zeros(self.plant.n_u)  

            #simulate the system
            while r.successful() and r.t < t_f:
                
                #define the control input. A random value is used for data generation
                r.set_f_params(u_t).set_jac_params(u_t)
                x_t = r.integrate(r.t + self.dt_data)             

                if r.t - t_control_last > self.dt_control:
                    t_control_last = r.t 
                    u_t =  np.random.uniform(low = self.plant.u_min , high = self.plant.u_max)

                #these are the inherent variables if the system ie input and state
                t_array.append(r.t)
                x_array.append(x_t)
                u_array.append([u_t])

                #these describe additional observations such as auxiliary variables or measurements
                eta_array.append(self.plant.phi(r.t,x_t,u_t))
                y_array.append(self.plant.g(r.t,x_t,u_t))

            eta_dot_array = np.gradient(np.array(eta_array),self.dt_data)[0]

            eta_dot_array2 = savgol_filter(np.array(eta_array),
                                           window_length = 5, polyorder = 3,
                                           deriv = 1, axis=0)/self.dt_data

            # if plot_sample and i == 0:
            #     fig, axs = plt.subplots(5, 1)
            #     axs[0].plot(np.array(t_array), np.array(x_array)[:,0], 'k')
            #     axs[1].plot(np.array(t_array), np.array(x_array)[:,1], 'k')
            #     axs[2].plot(np.array(t_array), np.array(x_array)[:,2], 'k')
            #     axs[3].plot(np.array(t_array), np.array(x_array)[:,3], 'k')
            #     axs[4].plot(np.array(t_array), np.array(u_array)[:,0], 'k')
                
            #     axs[4].set_xlabel('time')

            #     axs[0].set_ylabel('q1')
            #     axs[1].set_ylabel('q2')
            #     axs[2].set_ylabel('p1')
            #     axs[3].set_ylabel('p2')
            #     axs[4].set_ylabel('u')
                
            #     plt.show()

            Y_minus_data.append(y_array[:-1])
            U_minus_data.append(u_array[:-1])
            X_minus_data.append(x_array[:-1])
            Eta_minus_data.append(eta_array[:-1])

            Y_plus_data.append(y_array[1:])
            U_plus_data.append(u_array[1:])
            X_plus_data.append(x_array[1:])
            Eta_plus_data.append(eta_array[1:])


            t_data.append(t_array)
            x_data.append(x_array)
            u_data.append(u_array)
            eta_data.append(eta_array)
            eta_dot_data.append(eta_dot_array2)
        
        self.t_data = np.array(t_data) 
        self.x_data = np.array(x_data)
        self.u_data = np.array(u_data)
        self.eta_data = np.array(eta_data)
        self.eta_dot_data = np.array(eta_dot_data)

        self.Y_minus = np.array(Y_minus_data)
        self.U_minus = np.array(U_minus_data)
        self.X_minus = np.array(X_minus_data)
        self.Eta_minus = np.array(Eta_minus_data)

        self.Y_plus = np.array(Y_plus_data)
        self.U_plus = np.array(U_plus_data)
        self.X_plus = np.array(X_plus_data)
        self.Eta_plus = np.array(Eta_plus_data)

    def simulate_system_nonlinear(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.n_u,1))
        t,u,x,y = self.simulate_system(x_0, u_minus, t_f, self.dt_data,
                                        u_func, self.dt_control, self.plant.f, self.plant.g,
                                        continuous = True)
        
        return t, u, x, y
        
    def simulate_system_dfl(self, x_0, u_func, t_f, continuous = True):

        u_minus = np.zeros((self.plant.n_u,1))
        eta_0 = self.plant.phi(0.0, x_0, u_minus)
        xi_0 = np.concatenate((x_0,eta_0))
        
        if continuous == True:
            t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                            u_func, self.dt_control, self.f_cont_dfl, self.plant.g,
                                            continuous = True)
        else:
            t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                            u_func, self.dt_control, self.f_disc_dfl, self.plant.g,
                                            continuous = False)
            
        return t, u, xi, y 

    def simulate_system_koop(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.n_u,1))
        xi_0 = self.plant.g(0.0, x_0, u_minus)

        t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                        u_func,self.dt_control, self.f_disc_koop, self.plant.g,
                                        continuous = False)
        return t, u, xi, y 

    def simulate_system_hybrid(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.n_u,1))
        xi_0 = self.get_best_initial_hybrid(x_0, u_func, t_f)
        t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                        u_func, self.dt_control, self.f_disc_hybrid, self.g_disc_hybrid,
                                        continuous = False)

        return t,u,xi,y
    
    def get_best_initial_hybrid(self, x_0, u_func, t_f):

        u_0 = np.zeros((self.plant.n_u,1))
        eta_0 = self.plant.phi_hybrid(0.0, x_0, u_0)
        xi_0 = np.linalg.pinv(self.C_til).dot(eta_0 - self.D_til_1.dot(x_0) -self.D_til_2.dot(u_0)[:,0])

        z_0 = np.concatenate((x_0,xi_0))

        return z_0 

    def simulate_system_sid(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.n_u,1))
        xi_0 = np.linalg.pinv(self.C_disc_sid).dot(x_0)  #self.get_best_initial_sid(x_0, u_func, t_f)
        t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                        u_func, self.dt_control, self.f_disc_sid, self.g_disc_sid,
                                        continuous = False)
        return t,u,xi,y

    # def get_best_initial_sid(self, x_0, u_func, t_f):

    #     xi_0 = np.linalg.pinv(self.C_disc_sid).dot(x_0)

    #     return xi_0 
