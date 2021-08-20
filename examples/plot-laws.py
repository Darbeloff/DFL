#!/usr/bin/env python

import dfl.dynamic_system
import dfl.dynamic_model as dm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 30
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def phi_c1(q):
    # return np.sign(q)*q**2
    return -2*(1/(1+np.exp(-4*q))-0.5)

def phi_r1(er):
    return 2*(1/(1+np.exp(-4*er))-0.5)

def phi_c2(r):
    return -3*r+3*r**3

def phi_r2(er):
    return -3*er+3*er**3

def phi_i(p):
    return p**3

if __name__== "__main__":
    fig, axs = plt.subplots(2,2)

    x = np.linspace(-1.0,1.0)

    y_c1 = phi_c1(x)
    axs[0,0].plot(x,y_c1,'k')
    axs[0,0].set_xlabel(r'$q_1$')
    axs[0,0].set_ylabel(r'$\Phi_\mathrm{C}(q_1)$')

    y_r1 = phi_r1(x)
    axs[0,1].plot(x,y_r1,'k')
    axs[0,1].set_xlabel(r'$\mathrm{e_{R1}}$')
    axs[0,1].set_ylabel(r'$\Phi_\mathrm{R1}(\mathrm{e_{R1}})$')

    y_c2 = phi_c2(x)
    axs[1,0].plot(x,y_c2,'k')
    axs[1,0].set_xlabel(r'$q_2$, $\mathrm{f_{R2}}$')
    axs[1,0].set_ylabel(r'$\Phi_\mathrm{C1}(q_2)$, $\Phi_\mathrm{R2}(\mathrm{f_{R2}})$')

    y_i  = phi_i (x)
    axs[1,1].plot(x,y_i,'k')
    axs[1,1].set_xlabel(r'$p$')
    axs[1,1].set_ylabel(r'$\Phi_\mathrm{I}(p)$')

    plt.show()