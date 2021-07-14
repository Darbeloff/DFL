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
        self.A_cont_x   = np.array([[0.0]])
        self.A_cont_eta = np.array([[0.0, 1.0]])
        self.B_cont_x   = np.array([[0.0]])

        # Limits for inputs and states
        self.x_min = np.array([-2.0])
        self.x_max = np.array([ 2.0])
        self.u_min = np.array([-2.5])
        self.u_max = np.array([ 2.5])

    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c(q):
        return np.sign(q)*q**2

    @staticmethod
    def phi_r(er):
        return 2*(1/(1+np.exp(-4*er))-0.5)
    
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

def int_abs_error(y1, y2):
    # return np.sum(np.abs(y1-y2))
    return np.sum((y1-y2)**2)

if __name__== "__main__":
    driving_fun = dfl.dynamic_system.DFLDynamicPlant.sin_u_func
    plant1 = Plant1()
    tf = 4.0
    x_0 = np.zeros(plant1.n_x)
    fig, axs = plt.subplots(1, 1)

    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()
    t, u, x_tru, y_tru = tru.simulate_system(x_0, driving_fun, tf)
    axs.plot(t, u, 'gainsboro')
    axs.text(3.7, -0.43, 'u', fontsize='xx-large', color='tab:gray', fontstyle='italic')
    axs.plot(t, x_tru[:,0], 'k-', label='Gnd. Truth')

    koo = dm.Koopman(plant1, observable='polynomial')
    koo.learn(data)
    _, _, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, tf)
    axs.plot(t, x_koo[:,0], 'g-.', label='Koopman')
    print('Koopman Error: {}'.format(int_abs_error(x_koo[:,0],x_tru[:,0])))

    dmd = dm.Koopman(plant1, observable='polynomial', n_koop=4)
    dmd.learn(data, dmd=True)
    _, _, x_dmd, y_dmd = dmd.simulate_system(x_0, driving_fun, tf)
    axs.plot(t, x_dmd[:,0], 'c-.', label='DMDc')
    print('DMDc Error: {}'.format(int_abs_error(x_dmd[:,0],x_tru[:,0])))

    edmd = dm.Koopman(plant1, observable='polynomial')
    edmd.learn(data, dmd=True)
    _, _, x_edmd, y_edmd = edmd.simulate_system(x_0, driving_fun, tf)
    axs.plot(t, x_edmd[:,0], 'm-.', label='eDMDc')
    print('eDMDc Error: {}'.format(int_abs_error(x_edmd[:,0],x_tru[:,0])))

    dfl = dm.DFL(plant1, ac_filter=True)
    dfl.learn(data)
    _, _, x_dfl, y_dfl = dfl.simulate_system(x_0, driving_fun, tf)
    axs.plot(t, x_dfl[:,0], 'r-.', label='DFL')
    print('DFL Error: {}'.format(int_abs_error(x_dfl[:,0],x_tru[:,0])))

    # lrn = dm.L3(plant1, 2, ac_filter='linear', model_fn='model_toy_acf', retrain=False, hidden_units_per_layer=256, num_hidden_layers=2)
    # lrn.learn(data)
    # _, _, x_lrn, y_lrn = lrn.simulate_system(x_0, driving_fun, tf)
    # axs.plot(t, x_lrn[:,0], 'b-.', label='L3')
    # print('L3 Error: {}'.format(int_abs_error(x_lrn[:,0],x_tru[:,0])))

    # bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, fig.subplotpars.right-fig.subplotpars.left, .1)
    # axs.legend(bbox_to_anchor=bb, loc='lower left', ncol=3, mode="expand", borderaxespad=0., bbox_transform=fig.transFigure)

    axs.legend(ncol=3, loc='upper center')
    axs.set_xlabel('time (s)')
    axs.set_ylabel('x (m)')
    axs.set_ylim(-0.6, 1.2)
    
    plt.show()