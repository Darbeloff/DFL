#!/usr/bin/env python

import dfl.dynamic_system
import dfl.dynamic_model as dm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

class Plant1(dfl.dynamic_system.DFLDynamicPlant):
    def __init__(self):
        self.n_x = 1
        self.n_eta = 2
        self.n_u = 1

        # User defined matrices for DFL
        self.A_cont_x   = np.zeros((self.n_x,self.n_x))
        self.A_cont_eta = np.array([[1.0, 0.0]])
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
        # return 2*(1/(1+np.exp(-4*er))-0.5)
        return 3*er*(er+1)*(er-1)
    
    # nonlinear state equations
    @staticmethod
    def f(t,x,u):
        eta = Plant1.phi(t,x,u)
        return np.array([eta[0]])

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        return np.concatenate((np.copy(x), Plant1.phi(t,x,u)), 0)

    # auxiliary variables (outputs from nonlinear elements)
    @staticmethod
    def phi(t,x,u):
        q = x[0]

        ec = Plant1.phi_c(q)
        er = u[0]-ec
        f  = Plant1.phi_r(er)
        
        eta = np.array([f,ec])

        return eta

class Plant2(Plant1):
    def __init__(self):
        self.n_x = 1
        self.n_eta = 1
        self.n_u = 1

        self.assign_random_system_model()

    @staticmethod
    def phi(t,x,u):
        eta_old = Plant1.phi(t,x,u)

        e = eta_old[0]

        eta = np.array([e])

        return eta

    # @staticmethod
    # def f(t,x,u):
    #     eta_old = Plant1.phi(t,x,u)
    #     ec = eta_old[0]
    #     fl = eta_old[1]

    #     return np.array([fl])

    # @staticmethod
    # def g(t,x,u):
    #     return np.concatenate((np.copy(x), Plant2.phi(t,x,u)), 0)

def int_abs_error(y1, y2):
    # return np.sum(np.abs(y1-y2))
    return np.sum((y1-y2)**2)

if __name__== "__main__":
    driving_fun = dfl.dynamic_system.DFLDynamicPlant.sin_u_func
    tf = 2.0
    plant1 = Plant1()
    plant2 = Plant2()
    fig, axs = plt.subplots(plant1.n_x, 1)

    x_0 = np.zeros(plant1.n_x)
    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()
    t, u, x_tru, y_tru = tru.simulate_system(x_0, driving_fun, tf)
    for i in range(plant1.n_x):
        axs.plot(t, u, 'gainsboro')
        axs.text(1.7, -0.43, 'u', fontsize='xx-large', color='tab:gray', fontstyle='italic')
        axs.plot(t, y_tru[:,i], 'k-', label='Gnd. Truth')

    koo = dm.Koopman(plant1, observable='polynomial', n_koop=16)
    koo.learn(data)
    _, _, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, tf)
    for i in range(plant1.n_x): axs.plot(t, y_koo[:,i], linestyle='-.', color='blue', label='Koopman')
    print('Koopman Error: {}'.format(int_abs_error(x_koo[:,:2],x_tru)))

    dfl = dm.DFL(plant1, ac_filter=True)
    dfl.learn(data)
    _, _, x_dfl, y_dfl = dfl.simulate_system(x_0, driving_fun, tf)
    for i in range(plant1.n_x): axs.plot(t, x_dfl[:,i], 'r-.', label='DFL')
    print('DFL Error: {}'.format(int_abs_error(x_dfl[:,:2],x_tru)))

    x_0_2 = np.zeros(plant2.n_x)
    tru2 = dm.GroundTruth(plant2)
    data2 = tru2.generate_data_from_random_trajectories(x_0=x_0_2)

    idmd = dm.Koopman(plant2, observable='polynomial', n_koop=5)
    idmd.learn(data2, dmd=True)
    _, _, x_idmd, y_idmd = idmd.simulate_system(x_0_2, driving_fun, tf)
    for i in range(plant1.n_x): axs.plot(t, y_idmd[:,i], linestyle='-.', color='darkmagenta', label='iDMDc')
    print('iDMDc Error: {}'.format(int_abs_error(x_idmd[:,:2],x_tru)))

    # dfl = dm.DFL(plant2, ac_filter=True)
    # dfl.learn(data2)
    # _, _, x_dfl, y_dfl = dfl.simulate_system(x_0_2, driving_fun, tf)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs.plot(t, x_dfl[:,i], 'b-.', label='iDFL')
    # print('iDFL Error: {}'.format(int_abs_error(x_dfl[:,:2],x_tru)))

    # bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, fig.subplotpars.right-fig.subplotpars.left, .1)
    # axs.legend(bbox_to_anchor=bb, loc='lower left', ncol=3, mode="expand", borderaxespad=0., bbox_transform=fig.transFigure)

    # axs[0].legend(ncol=3, loc='upper center')
    axs.set_xlabel('time (s)')
    axs.set_ylabel('q')

    axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
    
    axs.set_ylim(-1,1)
    axs.set_ylim(-1,1)
    
    plt.show()