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
        self.n_x = 3
        self.n_eta = 5
        self.n_u = 1

        # User defined matrices for DFL
        self.A_cont_x   = np.zeros((self.n_x, self.n_x))
        self.A_cont_eta = np.array([[0.0,  0.0, -1.0, 1.0, -1.0],
                                    [1.0, -1.0,  0.0, 0.0,  0.0],
                                    [0.0,  1.0,  0.0, 0.0,  0.0]])
        self.B_cont_x   = np.zeros((self.n_x, self.n_u))

        # Limits for inputs and states
        self.x_min = -2.0*np.ones(self.n_x)
        self.x_max =  2.0*np.ones(self.n_x)
        self.u_min = -2.0*np.ones(self.n_u)
        self.u_max =  2.0*np.ones(self.n_u)

    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c1(q):
        # return np.sign(q)*q**2
        return -2*(1/(1+np.exp(-4*q))-0.5)

    @staticmethod
    def phi_r1(er):
        return 2*(1/(1+np.exp(-4*er))-0.5)

    @staticmethod
    def phi_c2(r):
        return -3*r+3*r**3

    @staticmethod
    def phi_r2(er):
        return -3*er+3*er**3

    @staticmethod
    def phi_i(p):
        return p**3

    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        if not isinstance(u,np.ndarray):
            u = np.array([u])

        p,q,r = np.ravel(x)

        fr1 = Plant1.phi_r1(u[0]-Plant1.phi_c1(q))
        fr2 = Plant1.phi_i(p)
        er2 = Plant1.phi_r2(fr2)
        ec1 = Plant1.phi_c1(q)
        ec2 = Plant1.phi_c2(r)

        return np.array([fr1, fr2, er2, ec1, ec2])
    
    # nonlinear state equations
    @staticmethod
    def f(t,x,u):
        p,q,r = np.ravel(x)

        pdot = Plant1.phi_c1(q) - Plant1.phi_r2(Plant1.phi_i(p)) - Plant1.phi_c2(r)
        qdot = Plant1.phi_r1(u[0]-Plant1.phi_c1(q)) - Plant1.phi_i(p)
        rdot = Plant1.phi_i(p)

        return np.array([pdot, qdot, rdot])

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        return np.copy(x)

class PlantI(Plant1):
    def __init__(self):
        self.n_x = 4
        self.n_eta = 4
        self.n_u = 1

        self.assign_random_system_model()

    def phi(self,t,x,u):
        if not isinstance(u,np.ndarray):
            u = np.array([u])

        p,q,r,fr1s = np.ravel(x)

        fr2 = Plant1.phi_i(p)
        er2 = Plant1.phi_r2(fr2)
        ec1 = Plant1.phi_c1(q)
        ec2 = Plant1.phi_c2(r)

        return np.array([fr2, er2, ec1, ec2])
    
    # nonlinear state equations
    @staticmethod
    def f(t,x,u):
        p,q,r,fr1s = np.ravel(x)

        pdot = Plant1.phi_c1(q) - Plant1.phi_r2(Plant1.phi_i(p)) - Plant1.phi_c2(r)
        qdot = Plant1.phi_r1(u[0]-Plant1.phi_c1(q)) - Plant1.phi_i(p)
        rdot = Plant1.phi_i(p)
        fr1sdot = Plant1.phi_r1(u[0]-Plant1.phi_c1(q))

        return np.array([pdot, qdot, rdot, fr1sdot])

class PlantAddM(Plant1):
    def __init__(self):
        self.n_x = 4
        self.n_eta = 6
        self.n_u = 1

        self.A_cont_x   = np.zeros((self.n_x, self.n_x))
        self.A_cont_eta = np.array([[0.0,  0.0, -1.0,  1.0, -1.0,  0.0],
                                    [1.0, -1.0,  0.0,  0.0,  0.0,  0.0],
                                    [0.0,  1.0,  0.0,  0.0,  0.0,  0.0],
                                    [0.0,  0.0,  0.0, -1.0,  0.0, -1.0]])
        self.B_cont_x   = np.array([[0.0],[0.0],[0.0],[1.0]])

        # Limits for inputs and states
        self.x_min = -2.0*np.ones(self.n_x)
        self.x_max =  2.0*np.ones(self.n_x)
        self.u_min = -2.0*np.ones(self.n_u)
        self.u_max =  2.0*np.ones(self.n_u)

    @staticmethod
    def phi_pa(p):
        return 10*p

    @staticmethod
    def phi(t,x,u):
        p,q1,q2,pa = np.ravel(x)

        fr1 = PlantAddM.phi_pa(pa)
        fr2 = Plant1.phi_i(p)
        er2 = Plant1.phi_r2(fr2)
        ec1 = Plant1.phi_c1(q1)
        ec2 = Plant1.phi_c2(q2)
        er1 = Plant1.phi_r1(fr1)

        return np.array([fr1, fr2, er2, ec1, ec2, er1])

    @staticmethod
    def f(t,x,u):
        fr1, fr2, er2, ec1, ec2, er1 = PlantAddM.phi(t,x,u)

        pdot = ec1-er2-ec2
        q1dot = fr1-fr2
        q2dot = fr2
        padot = u[0]-er1-ec1

        return np.array([pdot,q1dot,q2dot,padot])

def int_abs_error(y1, y2):
    # return np.sum(np.abs(y1-y2))
    return np.sum((y1-y2)**2)

def abs_error(y, tru):
    return np.sum(np.abs(y[:,:2]-tru[:,:2]),1)

def test_model(dm, x_0, driving_fun, tf, ls, color, label, tru):
    _, _, x, y = dm.simulate_system(x_0, driving_fun, tf)
    # err_sig = abs_error(y, tru)
    # axs.plot(t, err_sig, linestyle=ls, color=color, label=label)
    # print('{} Error: {}'.format(label, int_abs_error(y[:,:2],tru[:,:2])))
    return int_abs_error(y[:,:3],tru[:,:3])

if __name__== "__main__":
    # Setup
    tf = 10.0
    plant1 = Plant1()
    plant2 = PlantI()
    fig, axs = plt.subplots(1, 1)
    x_0 = np.zeros(plant1.n_x)
    x_0_2 = np.zeros(plant2.n_x)
    plantm = PlantAddM()
    x_0_m = np.zeros(plantm.n_x)
    
    # Training
    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()
    koo = dm.Koopman(plant1, observable='polynomial', n_koop=32)
    koo.learn(data, dmd=False)
    edc = dm.Koopman(plant1, observable='polynomial', n_koop=32)
    edc.learn(data, dmd=True)  
    adf = dm.DFL(plant1, ac_filter=True)
    adf.learn(data)
    # lrn = dm.L3(plant1, 8, ac_filter='linear', model_fn='model_exc', retrain=False, hidden_units_per_layer=256, num_hidden_layers=2)
    # lrn.learn(data)

    tru2 = dm.GroundTruth(plant2)
    data2 = tru2.generate_data_from_random_trajectories(x_0=x_0_2)
    idc = dm.Koopman(plant2, observable='polynomial')
    idc.learn(data2, dmd=True)
    
    trum = dm.GroundTruth(plantm)
    datam = trum.generate_data_from_random_trajectories()
    adc = dm.Koopman(plantm, observable='polynomial')
    adc.learn(datam, dmd=True)
    
    # Testing
    n_models = 6
    n_tests = 10
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

        err_arr[0,i] = test_model(koo, x_0  , driving_fun, tf, 'dotted' , 'g'          , 'KIC'  , y_tru)
        err_arr[1,i] = test_model(edc, x_0  , driving_fun, tf, 'dashed' , 'c'          , 'eDMDc', y_tru)
        err_arr[2,i] = test_model(adf, x_0  , driving_fun, tf, 'dashdot', 'r'          , 'DFL'  , y_tru)
        # err_arr[3,i] = test_model(lrn, x_0  , driving_fun, tf, 'dotted' , 'b'          , 'L3'   , y_tru)
        err_arr[4,i] = test_model(adc, x_0_m, driving_fun, tf, 'dashdot', 'chocolate'  , 'AL2', y_tru)
        err_arr[5,i] = test_model(idc, x_0_2, driving_fun, tf, 'dashed' , 'darkmagenta', 'IL2', y_tru)

    print(np.mean(err_arr,axis=1))
    print(np.std(err_arr, axis=1)/n_tests)