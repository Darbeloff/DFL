#!/usr/bin/env python
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

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
import itertools

import matplotlib.pyplot as plt

from dfl.LearnedDFL import LearnedDFL

np.set_printoptions(precision = 4)
np.set_printoptions(suppress = True)

H = 256
dtype = torch.FloatTensor
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
seed = 9
torch.manual_seed(seed)
np.random.seed(seed = seed)

RETRAIN = True
RETRAIN = False

def step(x_batch, y_batch, model, loss_fn):
    # Send data to GPU if applicable
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    x_t = y_batch

    eta_t = model.g(x_t)

    x_hat, eta_hat = model(x_batch)

    # Return 
    return loss_fn(x_t, x_hat) + loss_fn(eta_t, eta_hat)

class DFL():
    def __init__(self, dynamic_plant, dt_data = 0.05, dt_control = 0.1):
        
        self.plant = dynamic_plant
        self.dt_data = dt_data 
        self.dt_control = dt_control

    def train_model(self, model, x, y, title=None, opt='all'):
        # Reshape x and y to be vector of tensors
        x = torch.transpose(x,0,1)
        y = torch.transpose(y,0,1)

        dataset = TensorDataset(x, y)

        N_train = int(3*len(y)/5)
        train_dataset, val_dataset = random_split(dataset, [N_train,len(y)-N_train])

        train_loader = DataLoader(dataset=train_dataset, batch_size=50)
        val_loader   = DataLoader(dataset=val_dataset  , batch_size=50)

        loss_fn = torch.nn.MSELoss(reduction='sum')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algorithms. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        learning_rate = .0001
        n_epochs = 100000
        training_losses = []
        validation_losses = []
        # breakpoint()
        if opt is 'all':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(itertools.chain(model.A.parameters(), model.H.parameters()), lr=learning_rate)

        for t in range(n_epochs):
            batch_losses = []

            with torch.no_grad():
                val_losses = []
                for x_val, y_val in val_loader:
                    val_loss = step(x_val, y_val, model, loss_fn).item()
                    val_losses.append(val_loss)
                validation_loss = np.mean(val_losses)
                validation_losses.append(validation_loss)

            if t>100 and np.mean(validation_losses[-20:-11])<=np.mean(validation_losses[-10:-1]):
                break

            for x_batch, y_batch in train_loader:
                loss = step(x_batch, y_batch, model, loss_fn)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
            training_loss = np.mean(batch_losses)
            training_losses.append(training_loss)

            print(f"[{t+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

        
        plt.figure()
        plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
        plt.semilogy(range(len(validation_losses)), validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()

        model.eval()
        return model

    def learn_eta_fn(self):
        x_minus = torch.transpose(torch.from_numpy(self.X_minus.reshape(-1, self.X_minus.shape[-1])).type(dtype), 0,1)
        u_minus = torch.transpose(torch.from_numpy(self.U_minus.reshape(-1, self.U_minus.shape[-1])).type(dtype), 0,1)
        x_plus  = torch.transpose(torch.from_numpy(self.X_plus .reshape(-1, self.X_plus .shape[-1])).type(dtype), 0,1)

        self.model = LearnedDFL(self.plant.n_x, self.plant.n_eta, self.plant.n_u, 256)

        if RETRAIN:
            self.model = self.train_model(self.model, torch.cat((x_minus, u_minus), 0), x_plus)
            torch.save(self.model.state_dict(), 'model.pt')
        else:
            self.model.load_state_dict(torch.load('model.pt'))

        def eta_fn(x, u):
            x_shape = x.shape
            u_shape = u.shape
            if len(x_shape)==3:
                x = x.reshape(-1, x.shape[-1])
                u = u.reshape(-1, u.shape[-1])

            inp = torch.from_numpy(x).type(dtype)

            eta = self.model.g(inp).detach().numpy().T

            return eta

        self.eta_fn = eta_fn

    def regen_eta(self):
        self.EtaL_minus = self.eta_fn(self.X_minus, self.U_minus)
        self.EtaL_plus  = self.eta_fn(self.X_plus , self.U_plus )

    def lstsqAH(self):
        x_t   = torch.from_numpy(self.X_plus .reshape(-1, self.X_plus .shape[-1])).type(dtype)
        x_tm1 = torch.from_numpy(self.X_minus.reshape(-1, self.X_minus.shape[-1])).type(dtype)
        u_tm1 = torch.from_numpy(self.U_minus.reshape(-1, self.U_minus.shape[-1])).type(dtype)

        self.model.lstsqAH(x_t, x_tm1, u_tm1)

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
        omega = np.concatenate((self.  X_minus.reshape(-1, self.  X_minus.shape[-1]),
                                self.Eta_minus.reshape(-1, self.Eta_minus.shape[-1]),
                                self.  U_minus.reshape(-1, self.  U_minus.shape[-1])),axis=1).T
        
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

    def regress_New_Linear_Model(self):
        x_minus = torch.transpose(torch.from_numpy(self.X_minus.reshape(-1, self.X_minus.shape[-1])).type(dtype), 0,1)
        u_minus = torch.transpose(torch.from_numpy(self.U_minus.reshape(-1, self.U_minus.shape[-1])).type(dtype), 0,1)
        x_plus  = torch.transpose(torch.from_numpy(self.X_plus .reshape(-1, self.X_plus .shape[-1])).type(dtype), 0,1)

        self.model = self.train_model(self.model, torch.cat((x_minus, u_minus), 0), x_plus, opt='lin')

    def generate_sid_model(self,xi_order):

        U = self.U_minus.reshape(-1, self.U_minus.shape[-1]).T
        Y = self.X_minus.reshape(-1, self.X_minus.shape[-1]).T
        
        if len(Y.shape) == 1:
            Y=Y.T
            Y = np.expand_dims(Y, axis=0)

        method = 'N4SID'
        sys_id = system_identification(Y, U, method, SS_D_required = True, SS_fixed_order = xi_order)

        self.A_disc_sid,self.B_disc_sid,self.C_disc_sid,self.D_disc_sid = sys_id.A, sys_id.B, sys_id.C, sys_id.D

    # def generate_hybrid_model(self,xi_order):

    #     U = np.concatenate((self.X_minus.reshape(-1, self.X_minus.shape[-1]),
    #                         self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T

    #     Y_temp = self.Eta_minus.reshape(-1, self.Eta_minus.shape[-1]).T

    #     Y = self.plant.P.dot(Y_temp)
        
    #     if len(Y.shape) == 1:
    #         Y = Y.T
    #         Y = np.expand_dims(Y, axis=0)

    #     (A_disc_x, B_disc_x,_,_,_) = cont2discrete((self.plant.A_cont_x, self.plant.B_cont_x, 
    #                                             np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
    #                                             self.dt_data)
            
    #     (_,A_disc_eta_hybrid,_,_,_)   = cont2discrete((self.plant.A_cont_x, self.plant.A_cont_eta_hybrid, 
    #                                       np.zeros(self.plant.n_x), np.zeros(self.plant.n_eta)),
    #                                             self.dt_data)

    #     method = 'N4SID'
    #     sys_id = system_identification(Y, U, method, SS_D_required = True, SS_fixed_order = int(xi_order)) #, IC='AICc')#

    #     # SS_fixed_order = self.plant.N_eta,
    #     A_til,B_til,C_til,D_til = sys_id.A, sys_id.B, sys_id.C, sys_id.D
        
    #     B_til_1 = B_til[:,:self.plant.n_x]
    #     B_til_2 = B_til[:,self.plant.n_x:]
    #     D_til_1 = D_til[:,:self.plant.n_x]
    #     D_til_2 = D_til[:,self.plant.n_x:]

    #     A1 = A_disc_x + A_disc_eta_hybrid.dot(D_til_1)
    #     A2 = A_disc_eta_hybrid.dot(C_til)
    #     B1 = B_disc_x+ A_disc_eta_hybrid.dot(D_til_2)

    #     self.A_disc_hybrid_full =  np.block([[A1     ,  A2],
    #                                          [B_til_1,  A_til ]])

    #     self.B_disc_hybrid_full = np.block([[B1],
    #                                    [B_til_2]])

    #     self.C_til = C_til
    #     self.D_til_1 = D_til_1
    #     self.D_til_2 = D_til_2

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

    def f_disc_learned(self,t,x,u):
        if not isinstance(u, np.ndarray):
            u = np.array([u])

        # NumPy to PyTorch
        x = torch.from_numpy(x).type(dtype)
        u = torch.from_numpy(u).type(dtype)

        # Assemble into xi
        xi_t = torch.cat((x,u),0)

        # Propagate through model
        x_tp1 = self.model.A(xi_t)
        eta_tp1 = self.model.H(xi_t)

        # PyTorch to NumPy
        x_tp1   =   x_tp1.detach().numpy()
        eta_tp1 = eta_tp1.detach().numpy()

        return np.concatenate((x_tp1,eta_tp1))

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
    
    # def g_disc_hybrid(self,t,x,u):

    #     if not isinstance(u,np.ndarray):
    #         u = np.array([u])

    #     if len(u.shape) > 1:
    #         u = u.flatten()

    #     y = np.dot(self.C_disc_sid, x) + np.dot(self.D_disc_sid, u)
        
    #     return y
    
    def g_disc_sid(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        if len(u.shape) > 1:
            u = u.flatten()

        y = np.dot(self.C_disc_sid, x) + np.dot(self.D_disc_sid, u)
        
        return y
    
    # def g_disc_hybrid(self,t,x,u):

    #     if not isinstance(u,np.ndarray):
    #         u = np.array([u])

    #     if len(u.shape) > 1:
    #         u = u.flatten()

    #     eta = np.dot(self.C_til, x[self.plant.N_x:]) + np.dot(self.D_til_1, x[:self.plant.N_x]) + np.dot(self.D_til_2, u)

    #     y = np.concatenate((x[:self.plant.N_x],eta))

    #     return y

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

    def simulate_system_learned(self, x_0, u_func, t_f):
        u_minus = np.zeros((self.plant.n_u,1))
        eta_0 = self.eta_fn(x_0, u_minus)
        xi_0 = np.concatenate((x_0, eta_0))

        t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data, u_func, self.dt_control, self.f_disc_learned, self.plant.g, continuous=False)

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
