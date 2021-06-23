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
        return 2*(1/(1+np.exp(-4*er))-0.5)
        # return 3*er*(er+1)*(er-1)

    @staticmethod
    def phi_i(p):
        return p**3
    
    # nonlinear state equations
    def f(self,t,x,u):
        return np.array([Plant1.phi_c(x[1]), Plant1.phi_r(u[0]-Plant1.phi_c(x[1]))-Plant1.phi_i(x[0])])
        # return np.matmul(self.A_cont_x, x) + np.matmul(self.A_cont_eta, self.phi(t,x,u)) + np.matmul(self.B_cont_x, u)

    # nonlinear observation equations
    def g(self,t,x,u):
        return np.concatenate((np.copy(x), self.phi(t,x,u)), 0)

    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        if not isinstance(u,np.ndarray):
            u = np.array([u])
            
        p = x[0]
        q = x[1]

        e  = Plant1.phi_c(q)
        er = u.item()-e
        f  = Plant1.phi_r(er)
        fi = Plant1.phi_i(p)
        
        eta = np.array([e,f,fi])

        return eta

class Plant2(Plant1):
    def __init__(self):
        self.n_x = 3
        self.n_eta = 2
        self.n_u = 1

        self.assign_random_system_model()

    def phi(self,t,x,u):
        eta_old = super().phi(t,x,u)

        e = eta_old[0]
        fi = eta_old[2]

        eta = np.array([e,fi])

        return eta

    def f(self,t,x,u):
        p = x[0]
        q = x[1]

        eta_old = super().phi(t,x,u)
        e = eta_old[0]
        fl = eta_old[1]
        fi = eta_old[2]

        return np.array([e, fl-fi, fl])

    def g(self,t,x,u):
        return np.concatenate((np.copy(x), self.phi(t,x,u)), 0)

def int_abs_error(y1, y2):
    # return np.sum(np.abs(y1-y2))
    return np.sum((y1-y2)**2)

if __name__== "__main__":
    driving_fun = dfl.dynamic_system.DFLDynamicPlant.sin_u_func
    tf = 10.0#2.0
    PLOT_ALL = False
    plant1 = Plant1()
    plant2 = Plant2()
    fig, axs = plt.subplots(plant1.n_x+PLOT_ALL*plant1.n_eta, 1)

    x_0 = np.zeros(plant1.n_x)
    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()
    t, u, x_tru, y_tru = tru.simulate_system(x_0, driving_fun, tf)
    for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta):
        axs[i].plot(t, u, 'gainsboro')
        axs[i].text(1.7, -0.43, 'u', fontsize='xx-large', color='tab:gray', fontstyle='italic')
        axs[i].plot(t, y_tru[:,i], 'k-', label='Gnd. Truth')

    # koo = dm.Koopman(plant1, observable='polynomial')
    # koo.learn(data)
    # _, _, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, tf)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_koo[:,i], 'g-.', label='Koopman')
    # print('Koopman Error: {}'.format(int_abs_error(x_koo[:,0],x_tru[:,0])))

    x_0_2 = np.zeros(plant2.n_x)
    tru2 = dm.GroundTruth(plant2)
    data2 = tru2.generate_data_from_random_trajectories(x_0=x_0_2)
    # t, u, x_tru2, y_tru2 = tru2.simulate_system(x_0_2, driving_fun, tf)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_tru2[:,i], 'r-', label='Gnd. Truth')

    idmd = dm.Koopman(plant2, observable='polynomial', n_koop=5)
    idmd.learn(data2, dmd=True)
    _, _, x_idmd, y_idmd = idmd.simulate_system(x_0_2, driving_fun, tf)
    for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_idmd[:,i], linestyle='-.', color='darkmagenta', label='iDMDc')
    print('iDMDc Error: {}'.format(int_abs_error(x_idmd[:,:2],x_tru)))

    # dmd = dm.Koopman(plant1, observable='polynomial', n_koop=5)
    # dmd.learn(data, dmd=True)
    # _, _, x_dmd, y_dmd = dmd.simulate_system(x_0, driving_fun, tf)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_dmd[:,i], 'b-.', label='DMDc')
    # print('DMDc Error: {}'.format(int_abs_error(x_dmd[:,:2],x_tru)))

    eidmd = dm.Koopman(plant2, observable='polynomial', n_koop=64)
    eidmd.learn(data2, dmd=True)
    _, _, x_eidmd, y_eidmd = eidmd.simulate_system(x_0_2, driving_fun, tf)
    for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_eidmd[:,i], 'm-.', label='eiDMDc')
    print('eiDMDc Error: {}'.format(int_abs_error(x_eidmd[:,:2],x_tru)))

    edmd = dm.Koopman(plant1, observable='polynomial', n_koop=64)
    edmd.learn(data, dmd=True)
    _, _, x_edmd, y_edmd = edmd.simulate_system(x_0, driving_fun, tf)
    for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_edmd[:,i], 'b-.', label='eDMDc')
    print('eDMDc Error: {}'.format(int_abs_error(x_edmd[:,:2],x_tru)))

    dfl = dm.DFL(plant1, ac_filter=True)
    dfl.learn(data)
    _, _, x_dfl, y_dfl = dfl.simulate_system(x_0, driving_fun, tf)
    for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, x_dfl[:,i], 'r-.', label='DFL')
    print('DFL Error: {}'.format(int_abs_error(x_dfl[:,:2],x_tru)))

    # lrn = dm.L3(plant2, 8, ac_filter='none', model_fn='model', retrain=False, hidden_units_per_layer=256, num_hidden_layers=2)
    # lrn.learn(data2)
    # _, _, x_lrn, y_lrn = lrn.simulate_system(x_0_2, driving_fun, tf)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, x_lrn[:,i], 'g-.', label='L3')
    # print('L3 Error: {}'.format(int_abs_error(x_lrn[:,:2],x_tru)))

    # lnf = dm.L3(plant1, 2, ac_filter='none', model_fn='model_toy_nof', retrain=False, hidden_units_per_layer=256, num_hidden_layers=2)
    # lnf.learn(data)
    # _, _, x_lnf, y_lnf = lnf.simulate_system(x_0, driving_fun, tf)
    # axs.plot(t, x_lnf[:,0], 'm-.', label='L3 (NoF)')
    # print('L3 (NoF) Error: {}'.format(int_abs_error(x_lnf[:,0],x_tru[:,0])))

    # bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, fig.subplotpars.right-fig.subplotpars.left, .1)
    # axs.legend(bbox_to_anchor=bb, loc='lower left', ncol=3, mode="expand", borderaxespad=0., bbox_transform=fig.transFigure)

    axs[0].legend(ncol=3, loc='upper center')
    axs[1+PLOT_ALL*plant1.n_eta].set_xlabel('time (s)')
    axs[0].set_ylabel('p')
    axs[1].set_ylabel('q')
    if PLOT_ALL:
        axs[2].set_ylabel('e')
        axs[3].set_ylabel('f')
        axs[4].set_ylabel('fi')
    
    # axs[0].set_ylim(-1.5,1.5)
    # axs[1].set_ylim(-1.5,1.5)
    
    plt.show()