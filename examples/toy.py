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
        # return 2*(1/(1+np.exp(-4*er))-0.5)
        return 3*er*(er+1)*(er-1)

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

class Plant2(Plant1):
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
        return np.concatenate((np.copy(x), Plant2.phi(t,x,u)), 0)

class Plant3(Plant1):
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

        return np.array([e, fl-fi, fi])

    @staticmethod
    def g(t,x,u):
        return np.concatenate((np.copy(x), Plant3.phi(t,x,u)), 0)

def int_abs_error(y1, y2):
    # return np.sum(np.abs(y1-y2))
    return np.sum((y1-y2)**2)

def abs_error(y, tru):
    return np.sum(np.abs(y[:,:2]-tru[:,:2]),1)

if __name__== "__main__":
    driving_fun = dfl.dynamic_system.DFLDynamicPlant.sin_u_func
    tf = 2.0
    plant1 = Plant1()
    plant2 = Plant2()
    plant3 = Plant3()
    fig, axs = plt.subplots(1, 1)

    x_0 = np.zeros(plant1.n_x)
    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()
    t, u, x_tru, y_tru = tru.simulate_system(x_0, driving_fun, tf)
    err_sig = abs_error(y_tru, y_tru)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta):
        # axs[i].plot(t, u, 'gainsboro')
        # axs[i].text(1.7, -0.43, 'u', fontsize='xx-large', color='tab:gray', fontstyle='italic')
    axs.plot(t, err_sig, 'k-', label='Gnd. Truth')

    koo = dm.Koopman(plant1, observable='polynomial', n_koop=32)
    koo.learn(data, dmd=False)
    _, _, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, tf)
    err_sig = abs_error(y_koo, y_tru)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_koo[:,i], linestyle='-.', color='g', label='KIC')
    axs.plot(t, err_sig, linestyle='-.', color='g', label='KIC')
    print('KIC Error: {}'.format(int_abs_error(x_koo[:,:2],x_tru)))

    edc = dm.Koopman(plant1, observable='polynomial', n_koop=32)
    edc.learn(data, dmd=True)
    _, _, x_edc, y_edc = edc.simulate_system(x_0, driving_fun, tf)
    err_sig = abs_error(y_edc, y_tru)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_edc[:,i], linestyle='-.', color='c', label='eDMDc')
    axs.plot(t, err_sig, linestyle='-.', color='c', label='eDMDc')
    print('eDMDc Error: {}'.format(int_abs_error(x_edc[:,:2],x_tru)))

    dfl = dm.DFL(plant1, ac_filter=True)
    dfl.learn(data)
    _, _, x_dfl, y_dfl = dfl.simulate_system(x_0, driving_fun, tf)
    err_sig = abs_error(x_dfl, x_tru)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, x_dfl[:,i], 'r-.', label='DFL')
    axs.plot(t, err_sig, 'r-.', label='DFL')
    print('DFL Error: {}'.format(int_abs_error(x_dfl[:,:2],x_tru)))

    lrn = dm.L3(plant1, 8, ac_filter='linear', model_fn='model_toy', retrain=False, hidden_units_per_layer=256, num_hidden_layers=2)
    lrn.learn(data)
    _, _, x_lrn, y_lrn = lrn.simulate_system(x_0, driving_fun, tf)
    err_sig = abs_error(x_lrn, x_tru)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, x_lrn[:,i], 'b-.', label='L3')
    axs.plot(t, err_sig, 'b-.', label='L3')
    print('L3 Error: {}'.format(int_abs_error(x_lrn[:,0],x_tru[:,0])))

    x_0_2 = np.zeros(plant2.n_x)
    tru2 = dm.GroundTruth(plant2)
    data2 = tru2.generate_data_from_random_trajectories(x_0=x_0_2)

    idmd = dm.Koopman(plant2, observable='polynomial', n_koop=5)
    idmd.learn(data2, dmd=True)
    _, _, x_idmd, y_idmd = idmd.simulate_system(x_0_2, driving_fun, tf)
    err_sig = abs_error(y_idmd, y_tru)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, y_idmd[:,i], linestyle='-.', color='darkmagenta', label='iDMDc')
    axs.plot(t, err_sig, linestyle='-.', color='darkmagenta', label='iDMDc')
    print('iDMDc Error: {}'.format(int_abs_error(x_idmd[:,:2],x_tru)))

    # dfl = dm.DFL(plant2, ac_filter=True)
    # dfl.learn(data2)
    # _, _, x_dfl, y_dfl = dfl.simulate_system(x_0_2, driving_fun, tf)
    # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta): axs[i].plot(t, x_dfl[:,i], 'b-.', label='iDFL')
    # print('iDFL Error: {}'.format(int_abs_error(x_dfl[:,:2],x_tru)))

    # bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, fig.subplotpars.right-fig.subplotpars.left, .1)
    # axs.legend(bbox_to_anchor=bb, loc='lower left', ncol=3, mode="expand", borderaxespad=0., bbox_transform=fig.transFigure)

    # axs[0].legend(ncol=3, loc='upper center')
    axs.set_xlabel('time (s)')
    axs.set_ylabel('Error')
    axs.set_yscale('log')

    axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
    
    # axs.set_ylim(-1,1)
    
    plt.show()