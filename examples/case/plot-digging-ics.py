#!/usr/bin/env python

import dfl.dynamic_system
import dfl.dynamic_model as dm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams["font.family"] = "Times New Roman"
    
def generate_data_from_file(file_name: str, test_ndx: int=4):
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

if __name__== "__main__":
    test_ndx = 50
    fig, axs = plt.subplots(1,1)

    data, test_data = generate_data_from_file('data_nick_not_flat.npz', test_ndx=test_ndx)
    x0 = data['x']['data'][:,0,0]
    y0 = data['x']['data'][:,0,1]
    breakpoint()
    axs.plot(x0,y0,'*')
    plt.show()