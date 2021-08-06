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
        return -3*q*(q+1)*(q-1)
        # return np.sign(q)*q**2

    @staticmethod
    def phi_r(er):
        return 3*er*(er+1)*(er-1)
        # return 2*(1/(1+np.exp(-4*er))-0.5)
    
    # nonlinear state equations
    @staticmethod
    def f(t,x,u):
        x_dot = np.zeros(x.shape)
        eta = Plant1.phi(t,x,u)
        x_dot[0] = eta[1]

        return x_dot

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        return np.copy(x)

    # auxiliary variables (outputs from nonlinear elements)
    @staticmethod
    def phi(t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        if not isinstance(u,np.ndarray):
            u = np.array([u])
            
        q = x[0]
        ec = Plant1.phi_c(q)
        er = u[0]-ec
        f = Plant1.phi_r(er)
        
        eta = np.array([ec, f])

        return eta

class PlantI(Plant1):
    def __init__(self):
        self.n_x = 1
        self.n_eta = 1
        self.n_u = 1

        self.assign_random_system_model()

    @staticmethod
    def phi(t,x,u):
        return np.array([Plant1.phi_c(x[0])])

class PlantM(Plant1):
    def __init__(self):
        self.n_x = 2
        self.n_eta = 3
        self.n_u = 1

        # User defined matrices for DFL
        self.A_cont_x   = np.zeros((self.n_x, self.n_x))
        self.A_cont_eta = np.array([[ 0.0, 1.0,  0.0],
                                    [-1.0, 0.0, -1.0]])
        self.B_cont_x   = np.array([[0.0],
                                     1.0])

        # Limits for inputs and states
        self.x_min = -2.0*np.ones(self.n_x)
        self.x_max =  2.0*np.ones(self.n_x)
        self.u_min = -2.0*np.ones(self.n_u)
        self.u_max =  2.0*np.ones(self.n_u)

    @staticmethod
    def phi_i(p):
        return 1*p

    @staticmethod
    def phi(t,x,u):
        q,p = x

        ec = Plant1.phi_c(q)
        f = PlantM.phi_i(p)
        er = Plant1.phi_r(f)

        return np.array([ec, f, er])

    @staticmethod
    def f(t,x,u):
        ec,f,er = PlantM.phi(t,x,u)
        return np.array([f,u[0]-er-ec])

def int_abs_error(y1, y2):
    # return np.sum(np.abs(y1-y2))
    return np.sum((y1-y2)**2)

def test_model(dm, x_0, driving_fun, tf, ls, color, label, tru):
    _, _, x, y = dm.simulate_system(x_0, driving_fun, tf)
    # err_sig = abs_error(y, tru)
    # axs.plot(t, err_sig, linestyle=ls, color=color, label=label)
    # print('{} Error: {}'.format(label, int_abs_error(y[:,:1],tru[:,:1])))
    return int_abs_error(y[:,:1],tru[:,:1])

def cdf(data, bins=50):
    count, bins_count = np.histogram(data, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf

if __name__== "__main__":
    # Setup
    tf = 10.0
    plant1 = Plant1()
    planti = PlantI()
    plantm = PlantM()
    x_0 = np.zeros(plant1.n_x)
    x_0i = np.zeros(planti.n_x)
    x_0m = np.zeros(plantm.n_x)
    fig, axs = plt.subplots(1, 1)

    # Training
    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()

    trui = dm.GroundTruth(planti)
    datai = trui.generate_data_from_random_trajectories()
    idmdc = dm.Koopman(planti, observable='polynomial')
    idmdc.learn(datai)

    trum = dm.GroundTruth(plantm)
    datam = trum.generate_data_from_random_trajectories()
    mdmdc = dm.Koopman(plantm, observable='polynomial')
    mdmdc.learn(datam)

    # Testing
    n_models = 2
    n_tests = 1000
    err_arr = np.zeros((n_models, n_tests))
    tf=1.0
    for i in range(n_tests):
        driving_fun = np.random.normal(0.0,0.3,(int(tf/dm.DT_CTRL_DEFAULT),1))
        t, u, x_tru, y_tru = tru.simulate_system(x_0, driving_fun, tf)
        # err_sig = abs_error(y_tru, y_tru)
        # for i in range(plant1.n_x+PLOT_ALL*plant1.n_eta):
            # axs[i].plot(t, u, 'gainsboro')
            # axs[i].text(1.7, -0.43, 'u', fontsize='xx-large', color='tab:gray', fontstyle='italic')
        # axs.plot(t, err_sig, 'k-', label='Gnd. Truth')

        err_arr[0,i] = test_model(mdmdc, x_0m, driving_fun, tf, 'dashdot', 'chocolate'  , 'aDMDc', y_tru)
        err_arr[1,i] = test_model(idmdc, x_0i, driving_fun, tf, 'dashed' , 'darkmagenta', 'iDMDc', y_tru)

    print('Mean', np.mean(err_arr,axis=1))
    print('stderr', np.std(err_arr, axis=1)/n_tests)
    print('median', np.median(err_arr, axis=1))
    print('90th p', np.percentile(err_arr, 90, axis=1))

    xa, cdfa = cdf(err_arr[0,:])
    xi, cdfi = cdf(err_arr[1,:])
    axs.semilogx(xa, cdfa, color='chocolate', label='AL2')
    axs.semilogx(xi, cdfi, color='darkmagenta', label='IL2')

    # axs.legend(ncol=2, loc='upper center')
    axs.legend()
    axs.set_xlabel('SSE')
    axs.set_ylabel('CDF')
    # axs.set_ylim(-0.6, 1.2)
    
    plt.show()