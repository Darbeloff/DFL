#!/usr/bin/env python

import dfl.dynamic_system
import dfl.dynamic_model as dm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

plt.rcParams["font.family"] = "Times New Roman"

class Plant1(dfl.dynamic_system.DFLDynamicPlant):
    
    def __init__(self):
        self.n_x = 6
        self.n_eta = 3
        self.n_u = 3

        self.assign_random_system_model()

    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c(q):
        return np.sign(q)*q**2

    @staticmethod
    def phi_r(f):
        return 2*(1/(1+np.exp(-4*f))-0.5)
    
    # nonlinear state equations
    def f(self,t,x,u):
        pass

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        return np.copy(x)

    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        pass
    
    @staticmethod
    def generate_data_from_file(file_name: str, test_ndx: int=4, truncate: float=0):
        '''
        x = [x, y, phi, v_x, v_y, omega], e = [a_x,a_y,alpha, F_x, F_y, m_soil]
        u = [u_x,u_y,tau]
        '''

        # Extract data from file
        data = np.load(file_name)
        t = data['t']
        x = data['x']
        e = data['e']
        e = e[:,:,3:] # Filippos Curating: rm accelerations
        u = data['u']

        # Truncate
        last_ind = int((1-truncate)*len(t))
        t = t[:last_ind]
        x = x[:last_ind]
        e = e[:last_ind]
        u = u[:last_ind]

        # Assemble data into paradigm
        t_data = t
        x_data = x
        u_data = u
        eta_data = e
        eta_dot_data = e
        y_data = np.copy(x)

        # Set aside test data
        t_data_test       = np.copy(t_data[test_ndx])
        x_data_test       = np.copy(x_data[test_ndx])
        u_data_test       = np.copy(u_data[test_ndx])
        eta_data_test     = np.copy(eta_data[test_ndx])
        eta_dot_data_test = np.copy(eta_dot_data[test_ndx])
        y_data_test       = np.copy(y_data[test_ndx])

        # Remove test data from training data
        t_data       = np.delete(      t_data,test_ndx,0)
        x_data       = np.delete(      x_data,test_ndx,0)
        u_data       = np.delete(      u_data,test_ndx,0)
        eta_data     = np.delete(    eta_data,test_ndx,0)
        eta_dot_data = np.delete(eta_dot_data,test_ndx,0)
        y_data       = np.delete(      y_data,test_ndx,0)

        # Inputs
        y_minus   = np.copy(  y_data[:, :-1,:])
        u_minus   =           u_data[:, :-1,:]
        x_minus   =           x_data[:, :-1,:]
        eta_minus =         eta_data[:, :-1,:]

        # Outputs
        y_plus   = np.copy(  y_data[:,1:  ,:])
        u_plus   =           u_data[:,1:  ,:]
        x_plus   =           x_data[:,1:  ,:]
        eta_plus =         eta_data[:,1:  ,:]

        # Return
        data = {
            't': t_data,
            'u': {
                'data':  u_data,
                'minus': u_minus,
                'plus':  u_plus
            },
            'x': {
                'data':  x_data,
                'minus': x_minus,
                'plus':  x_plus
            },
            'y': {
                'data':  y_data,
                'minus': y_minus,
                'plus':  y_plus
            },
            'eta': {
                'data':  eta_data,
                'minus': eta_minus,
                'plus':  eta_plus
            },
            'eta_dot': {
                'data':  eta_dot_data
            }
        }
        test_data = {
            't': t_data_test,
            'u': u_data_test,
            'x': x_data_test,
            'y': y_data_test,
            'eta': eta_data_test,
            'eta_dot': eta_dot_data_test
        }
        return data, test_data

def main(test_ndx):
    plant1 = Plant1()
    fig, axs = plt.subplots(2,3)

    data, test_data = Plant1.generate_data_from_file('data_nick_not_flat.npz', test_ndx=test_ndx)
    driving_fun = test_data['u']
    t = test_data['t']
    dt_data = t[1]-t[0]
    dt_control = t[1]-t[0]
    x_0 = np.copy(test_data['x'  ][0,:])
    z_0 = np.copy(test_data['eta'][0,:])
    xs_0 = np.concatenate((x_0,z_0))
    axs[0,0].plot(t, test_data['x'  ][:,0], 'k-', label='Ground Truth') # x
    axs[0,1].plot(t, test_data['x'  ][:,1], 'k-')   # y
    axs[0,2].plot(t, test_data['x'  ][:,2], 'k-')   # phi
    axs[1,0].plot(t, test_data['eta'][:,0], 'k-')   # Fx
    axs[1,1].plot(t, test_data['eta'][:,1], 'k-')   # Fx
    axs[1,2].plot(t, test_data['eta'][:,2], 'k-')   # m

    koo = dm.Koopman(plant1, dt_data=dt_data, dt_control=dt_control, observable='polynomial', n_koop=64)
    start_time = time.time()
    koo.learn(data)
    print('Koopman: {}'.format(time.time()-start_time))
    t, u, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_koo[:,0], 'g-.', label='Koopman')
    axs[0,1].plot(t, x_koo[:,1], 'g-.')
    axs[0,2].plot(t, x_koo[:,2], 'g-.')

    dmd = dm.Koopman(plant1, dt_data=dt_data, dt_control=dt_control, observable='polynomial', n_koop=len(xs_0))
    start_time = time.time()
    dmd.learn(data, dmd=True)
    print('DMDc: {}'.format(time.time()-start_time))
    _, _, x_dmd, y_dmd = dmd.simulate_system(xs_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_dmd[:,0], 'r-.', label='DMDc')
    axs[0,1].plot(t, x_dmd[:,1], 'r-.')
    axs[0,2].plot(t, x_dmd[:,2], 'r-.')
    axs[1,0].plot(t, x_dmd[:,6], 'r-.')
    axs[1,1].plot(t, x_dmd[:,7], 'r-.')
    axs[1,2].plot(t, x_dmd[:,8], 'r-.')

    lrn = dm.L3(plant1, 4, dt_data=dt_data, dt_control=dt_control, ac_filter='linear', retrain=False, model_fn='model_dig', hidden_units_per_layer=64)
    start_time = time.time()
    lrn.learn(data)
    print('L3: {}'.format(time.time()-start_time))
    _, _, x_lrn, y_lrn = lrn.simulate_system(xs_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_lrn[:,0], 'b-.', label='L3')
    axs[0,1].plot(t, x_lrn[:,1], 'b-.')
    axs[0,2].plot(t, x_lrn[:,2], 'b-.')
    axs[1,0].plot(t, x_lrn[:,6], 'b-.')
    axs[1,1].plot(t, x_lrn[:,7], 'b-.')
    axs[1,2].plot(t, x_lrn[:,8], 'b-.')

    lnf = dm.L3(plant1, 4, dt_data=dt_data, dt_control=dt_control, ac_filter='none', retrain=False, model_fn='model_dig_nof', hidden_units_per_layer=64)
    start_time = time.time()
    lnf.learn(data)
    print('L3 NoF: {}'.format(time.time()-start_time))
    _, _, x_lnf, y_lnf = lnf.simulate_system(xs_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_lnf[:,0], 'm-.', label='L3 (NoF)')
    axs[0,1].plot(t, x_lnf[:,1], 'm-.')
    axs[0,2].plot(t, x_lnf[:,2], 'm-.')
    axs[1,0].plot(t, x_lnf[:,6], 'm-.')
    axs[1,1].plot(t, x_lnf[:,7], 'm-.')
    axs[1,2].plot(t, x_lnf[:,8], 'm-.')

    lnz = dm.L3(plant1, 4, dt_data=dt_data, dt_control=dt_control, ac_filter='none', retrain=False, model_fn='model_dig_noz', ignore_zeta=True, hidden_units_per_layer=64)
    start_time = time.time()
    lnz.learn(data)
    print('L3 NoZ: {}'.format(time.time()-start_time))
    _, _, x_lnz, y_lnz = lnz.simulate_system(x_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_lnz[:,0], 'y-.', label='L3 (NoZ)')
    axs[0,1].plot(t, x_lnz[:,1], 'y-.')
    axs[0,2].plot(t, x_lnz[:,2], 'y-.')

    bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, fig.subplotpars.right-fig.subplotpars.left, .1)
    axs[0,0].legend(bbox_to_anchor=bb, loc='lower left', ncol=6, mode="expand", borderaxespad=0., bbox_transform=fig.transFigure)

    for c in range(3):
        axs[1,c].set_xlabel('time (s)')

    axs[0,0].set_ylabel('x (m)')
    axs[0,1].set_ylabel('y (m)')
    axs[0,2].set_ylabel('phi (rad)')
    axs[1,0].set_ylabel('F_x (kN)')
    axs[1,1].set_ylabel('F_y (kN)')
    axs[1,2].set_ylabel('m_soil (MT)')

    for c in range(3):
        axs[0,c].set_xlim(-0.1,2)
    axs[0,0].set_ylim(-3.4,-2.2)
    axs[0,1].set_ylim( 0.1, 0.7)
    axs[0,2].set_ylim( 1.1, 2.3)

    plt.show()

if __name__== "__main__":
    # for test_ndx in range(3,6):
    #     main(test_ndx)
    main(5)