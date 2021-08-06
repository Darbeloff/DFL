from abc import ABC, abstractmethod
import numpy as np
import scipy
from scipy import signal

class DFLDynamicPlant(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray, t: float):
        '''
        This function defines the time transition of a continuous time dynamic system.
        x: state
        u: exogenous input
        t: time
        '''
        pass

    @abstractmethod
    def g(self, x: np.ndarray, u: np.ndarray, t: float):
        '''
        This function defines the observed output of the system.
        x: state
        u: exogenous input
        t: time
        '''
        pass

    @abstractmethod
    def phi(self, x: np.ndarray, u: np.ndarray, t: float):
        '''
        outputs the values of the auxiliary variables
        '''
        pass

    def assign_random_system_model(self):
        self.A_cont_x   = np.random.rand(self.n_x, self.n_x  )
        self.A_cont_eta = np.random.rand(self.n_x, self.n_eta)
        self.B_cont_x   = np.random.rand(self.n_x, self.n_u  )

        # Limits for inputs and states
        self.x_min = -np.random.rand(self.n_x)
        self.x_max =  np.random.rand(self.n_x)
        self.u_min = -np.random.rand(self.n_u)
        self.u_max =  np.random.rand(self.n_u)

    @staticmethod
    def zero_u_func(y,t):
        return np.array([1]) 

    @staticmethod
    def rand_u_func(y,t):
        return np.random.normal(0.0,0.3,1)

    @staticmethod
    def sin_u_func(y,t):
        return np.array([0.5*scipy.signal.square(3*t)])