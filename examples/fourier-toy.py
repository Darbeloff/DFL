#!/usr/bin/env python

from os import stat
import dfl.dynamic_system
import dfl.dynamic_model as dm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 36
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
lw = 5

class Plant1(dfl.dynamic_system.DFLDynamicPlant):
    def __init__(self):
        self.n_x = 2
        self.n_eta = 3
        self.n_u = 1

        # User defined matrices for DFL
        self.A_cont_x   = np.zeros((self.n_x,self.n_x))
        self.A_cont_eta = np.array([[1.0, 0.0,  0.0],
                                    [0.0, 1.0, -1.0]])
        self.B_cont_x   = np.zeros((self.n_x,self.n_u))

        # Limits for inputs and states
        self.x_min = -2.0*np.ones(self.n_x)
        self.x_max =  2.0*np.ones(self.n_x)
        self.u_min = -2.0*np.ones(self.n_u)
        self.u_max =  2.0*np.ones(self.n_u)

    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c(q):
        return np.sign(q)*q**2

    @staticmethod
    def phi_r(er):
        return np.sign(er)*er**4
        # return (1/(1+np.exp(-4*er))-0.5)
        # return er*(er+1)*(er-1)
        # return Plant1.phi_c(er)+Plant1.phi_i(er)

    @staticmethod
    def phi_i(p):
        return p**3
    
    # nonlinear state equations
    @staticmethod
    def f(t,x,u):
        return np.array([Plant1.phi_c(x[1]), Plant1.phi_r(u[0]-Plant1.phi_c(x[1]))-Plant1.phi_i(x[0])])
        # return np.matmul(self.A_cont_x, x) + np.matmul(self.A_cont_eta, self.phi(t,x,u)) + np.matmul(self.B_cont_x, u)

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        return np.concatenate((np.copy(x), Plant1.phi(t,x,u)), 0)

    # auxiliary variables (outputs from nonlinear elements)
    @staticmethod
    def phi(t,x,u):
        # if not isinstance(u,np.ndarray):
        #     u = np.array([u])
            
        p = x[0]
        q = x[1]

        e  = Plant1.phi_c(q)
        er = u[0]-e
        f  = Plant1.phi_r(er)
        fi = Plant1.phi_i(p)
        
        eta = np.array([e,f,fi])

        return eta

class PlantI(Plant1):
    def __init__(self):
        self.n_x = 3
        self.n_eta = 2
        self.n_u = 1

        self.assign_random_system_model()

    @staticmethod
    def phi(t,x,u):
        eta_old = Plant1.phi(t,x,u)

        e = eta_old[0]
        fi = eta_old[2]

        eta = np.array([e,fi])

        return eta

    @staticmethod
    def f(t,x,u):
        eta_old = Plant1.phi(t,x,u)
        e = eta_old[0]
        fl = eta_old[1]
        fi = eta_old[2]

        return np.array([e, fl-fi, fl])

    @staticmethod
    def g(t,x,u):
        return np.concatenate((np.copy(x), PlantI.phi(t,x,u)), 0)

class PlantAddM(Plant1):
    def __init__(self):
        self.n_x = 3
        self.n_eta = 4
        self.n_u = 1

        # User defined matrices for DFL
        self.A_cont_x   = np.zeros((self.n_x,self.n_x))
        self.A_cont_eta = np.array([[ 1.0, 0.0,  0.0, 0.0 ],
                                    [ 0.0, 1.0, -1.0, 0.0 ],
                                    [-1.0, 0.0,  0.0, -1.0]])
        self.B_cont_x   = np.array([[0.0],
                                    [0.0],
                                    [1.0]])

        # Limits for inputs and states
        self.x_min = -2.0*np.ones(self.n_x)
        self.x_max =  2.0*np.ones(self.n_x)
        self.u_min = -2.0*np.ones(self.n_u)
        self.u_max =  2.0*np.ones(self.n_u)

    @staticmethod
    def phi_i0(p):
        return 10*p

    @staticmethod
    def phi(t,x,u):
        p,q,p0 = x

        e = Plant1.phi_c(q)
        f = PlantAddM.phi_i0(p0)
        fi = plant1.phi_i(p)
        er = plant1.phi_r(f)

        return np.array([e,f,fi,er])

    @staticmethod
    def f(t,x,u):
        e,f,fi,er = PlantAddM.phi(t,x,u)
        return np.array([e,f-fi,u[0]-e-er])

    @staticmethod
    def g(t,x,u):
        return np.concatenate((np.copy(x), PlantAddM.phi(t,x,u)), 0)

def int_abs_error(y1, y2):
    # return np.sum(np.abs(y1-y2))
    return np.sum((y1-y2)**2)

def abs_error(y, tru):
    # return np.sum(np.abs(y[:,:2]-tru[:,:2]),1)
    return np.sum((y-tru)**2,1)

if __name__== "__main__":
    # Setup
    tf = 10.0
    plant1 = Plant1()
    planti = PlantI()
    plantm = PlantAddM()
    x_0 = np.zeros(plant1.n_x)
    x_0i = np.zeros(planti.n_x)
    x_0m = np.zeros(plantm.n_x)
    
    # Training
    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()
    koo = dm.Koopman(plant1, observable='polynomial', n_koop=8)
    koo.learn(data, dmd=False)
    edc = dm.Koopman(plant1, observable='polynomial', n_koop=8)
    edc.learn(data, dmd=True)
    fko = dm.Koopman(plant1, observable='fourier', n_koop=8)
    fko.learn(data, dmd=False)
    fed = dm.Koopman(plant1, observable='fourier', n_koop=8)
    fed.learn(data, dmd=True)  

    tru2 = dm.GroundTruth(planti)
    data2 = tru2.generate_data_from_random_trajectories(x_0=x_0i)
    idc = dm.Koopman(planti, n_koop=planti.n_x+planti.n_eta)
    idc.learn(data2, dmd=True)
    
    trum = dm.GroundTruth(plantm)
    datam = trum.generate_data_from_random_trajectories(x_0=x_0m)
    adc = dm.Koopman(plantm, n_koop=plantm.n_x+plantm.n_eta)
    adc.learn(datam, dmd=True)
    
    # Testing
    driving_fun = dfl.dynamic_system.DFLDynamicPlant.sin_u_func
    tf = 1.0
    # fig, axs = plt.subplots(plant1.n_x, 1)
    fig, axs = plt.subplots(1, 1)

    def test_model(dm, x_0, driving_fun, tf, ls, color, label, tru):
        t, _, x, y = dm.simulate_system(x_0, driving_fun, tf)

        err_sig = abs_error(y[:,:plant1.n_x], tru)
        axs.plot(t, err_sig, linestyle=ls, color=color, linewidth=lw, label=label)
        print('{} Error: {}'.format(label, int_abs_error(y[:,:plant1.n_x],tru[:,:plant1.n_x])))

        # for i in range(plant1.n_x): axs[i].plot(t, y[:,i], linestyle=ls, color=color, label=label)
        # print('{} Error: {}'.format(label, int_abs_error(x[:,:plant1.n_x],tru)))

    t, u, x_tru, y_tru = tru.simulate_system(x_0, driving_fun, tf)
    # for i in range(plant1.n_x):
    #     axs[i].plot(t, u, 'gainsboro')
    #     axs[i].text(1.7, -0.43, 'u', fontsize='xx-large', color='tab:gray', fontstyle='italic')
    #     axs[i].plot(t, x_tru[:,i], 'k-', label='Gnd. Truth')

    test_model(koo,x_0,driving_fun,tf,'--','g','Poly. KIC',x_tru)
    test_model(edc,x_0,driving_fun,tf,':','c','Poly. eDMDc',x_tru)
    test_model(fko,x_0,driving_fun,tf,'-.','r','Fourier KIC',x_tru)
    test_model(fed,x_0,driving_fun,tf,'--','b','Fourier eDMDc',x_tru)
    test_model(adc,x_0,driving_fun,tf,':','chocolate','AL2',x_tru)
    test_model(idc,x_0,driving_fun,tf,'-.','darkmagenta','IL2',x_tru)

    # bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, fig.subplotpars.right-fig.subplotpars.left, .1)
    # axs.legend(bbox_to_anchor=bb, loc='lower left', ncol=3, mode="expand", borderaxespad=0., bbox_transform=fig.transFigure)

    # axs[0].legend(ncol=3, loc='upper center')
    axs.set_xlabel('time (s)')
    axs.set_ylabel('Error')
    axs.set_yscale('log')
    axs.set_xlim(0,1)

    # axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
    axs.legend(ncol=3, loc='lower right')
    
    # axs.set_ylim(-1,1)
    
    plt.show()