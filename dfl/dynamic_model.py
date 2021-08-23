#!/usr/bin/env python

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
import numpy.random
import itertools, copy, torch, scipy
from scipy import integrate, signal
from enum import Enum
import matplotlib.pyplot as plt

import dfl.dynamic_system
import dfl.L3Module as L3Module

np.set_printoptions(precision = 4)
np.set_printoptions(suppress = True)

dtype = torch.FloatTensor
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
seed = 1
torch.manual_seed(seed)
np.random.seed(seed = seed)
# torch.autograd.set_detect_anomaly(True)
# torch.set_num_threads(8)

DT_DATA_DEFAULT = 0.05
DT_CTRL_DEFAULT = 0.1

SCALE = 0.03

class DynamicModel(ABC):
    def __init__(self, dynamic_plant: dfl.dynamic_system.DFLDynamicPlant, dt_data: float=DT_DATA_DEFAULT, dt_control: float=DT_CTRL_DEFAULT, name: str=''):
        self.plant = dynamic_plant
        self.dt_data = dt_data
        self.dt_control = dt_control
        self.name = name
        self.trained = False

    def simulate_system(self, x_0: np.ndarray, u_minus: np.ndarray, t_f: float, u_func, f_func: Callable, g_func: Callable, continuous: bool=True):
        '''
        Simulate a system in continuous time
        Arguments:
        x_0: initial state
        u_func: control function or time series array
        t_f: final time
        '''

        # initial time and input
        t = 0.0

        # create numerical integration object
        if continuous:
            r = scipy.integrate.ode(f_func).set_integrator('vode', method = 'bdf', max_step = 0.001)
            r.set_initial_value(x_0,t)

        t_array = []
        x_array = []
        u_array = []
        y_array = []
        
        # initial state and 
        x_t = np.copy(x_0)
        y_t = g_func(t, x_t, u_minus)

        if callable(u_func):
            u_t = u_func(y_t, t)
        else:
            u_t = u_func[0]

        t_array.append(t)
        x_array.append(x_t)
        u_array.append(u_t)
        y_array.append(g_func(t,x_t,u_minus))

        t_control_last = 0
        i_u = 1
        #Simulate the system
        while t < t_f:

            if continuous:
                r.set_f_params(u_t).set_jac_params(u_t)
                x_t = r.integrate(r.t + self.dt_data)
            else:
                x_t = f_func(t, x_t, u_t)

            # NICK HACK XXX
            x_t+= np.random.normal(scale=SCALE,size=np.shape(x_t))
            t = t + self.dt_data
            y_t = g_func(t, x_t, u_t)
            
            if t - t_control_last > self.dt_control:
                t_control_last = copy.copy(t)
                if callable(u_func):
                    u_t = u_func(g_func(t, x_t, u_t), t)
                else:
                    u_t = u_func[i_u]
                    i_u+= 1
            # breakpoint()
            t_array.append(t)
            x_array.append(x_t)
            u_array.append(u_t)
            y_array.append(y_t)

        return np.array(t_array), np.array(u_array), np.array(x_array), np.array(y_array)

    def check_for_training(self):
        assert self.trained, 'Must train %s model before simulating. Run `model.learn(data)`.' %self.name

    @abstractmethod
    def learn(self, data: np.ndarray):
        pass

    @abstractmethod
    def f(self,t: float, x: np.ndarray, u:np.ndarray):
        pass

    @staticmethod
    def copy_into_minus_plus(data: np.ndarray):
        minus = np.copy(np.array(data)[:, :-1,:])
        plus  = np.copy(np.array(data)[:,1:  ,:])
        return minus, plus

    @staticmethod
    def lift_space(data: np.ndarray, g: Callable):
        y_data = []
        for traj in data:
            y = []
            for x in traj:
                y.append(g(x))
            y_data.append(y)
        y_minus, y_plus = DynamicModel.copy_into_minus_plus(y_data)

        return np.array(y_data), y_minus, y_plus

    @staticmethod
    def flatten_trajectory_data(data: np.ndarray):
        return data.reshape(-1, data.shape[-1])

class GroundTruth(DynamicModel):
    def __init__(self, dynamic_plant, dt_data=DT_DATA_DEFAULT, dt_control=DT_CTRL_DEFAULT, name='Ground Truth'):
        super().__init__(dynamic_plant, dt_data, dt_control, name)

    def simulate_system(self, x_0, u_func, t_f):
        u_minus = np.zeros((self.plant.n_u,1))
        t,u,x,y = super().simulate_system(x_0, u_minus, t_f, u_func, self.plant.f, self.plant.g, continuous=True)
        
        return t, u, x, y

    def generate_data_from_random_trajectories(self, t_range_data=10.0, n_traj_data=50, x_0=None):
        '''
        create random data to train DFL and other dynamic system models
        '''
        # Random input function
        u_func = lambda y,t : np.random.uniform(low=self.plant.u_min, high=self.plant.u_max)

        # Generate n_traj_data random trajectories using simulate_system
        t_data = []
        u_data = []
        x_data = []
        y_data = []
        eta_data = []
        eta_dot_data = []
        for _ in range(n_traj_data):
            # Randomly initialize state if one is not given
            if x_0 is None:
                x_0 = np.random.uniform(self.plant.x_min,self.plant.x_max)

            t, u, x, y = self.simulate_system(x_0, u_func, t_range_data)

            eta  = []
            for i in range(len(t)):
                eta.append(self.plant.phi(t[i],x[i],u[i]))
            # eta_dot = np.gradient(np.array(eta),self.dt_data)[0]
            eta_dot = scipy.signal.savgol_filter(np.array(eta), window_length = 5, polyorder = 3, deriv = 1, axis=0)/self.dt_data

            t_data.append(t)
            u_data.append(u)
            x_data.append(x)
            y_data.append(y)
            eta_data.append(eta)
            eta_dot_data.append(eta_dot)

        # Format data for output
        t_data       = np.array(      t_data) 
        u_data       = np.array(      u_data)
        x_data       = np.array(      x_data)
        y_data       = np.array(      y_data)
        eta_data     = np.array(    eta_data)
        eta_dot_data = np.array(eta_dot_data)

        u_minus, u_plus             = DynamicModel.copy_into_minus_plus(      u_data)
        x_minus, x_plus             = DynamicModel.copy_into_minus_plus(      x_data)
        y_minus, y_plus             = DynamicModel.copy_into_minus_plus(      y_data)
        eta_minus, eta_plus         = DynamicModel.copy_into_minus_plus(    eta_data)
        eta_dot_minus, eta_dot_plus = DynamicModel.copy_into_minus_plus(eta_dot_data)
        
        return {
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
                'data':  eta_dot_data,
                'minus': eta_dot_minus,
                'plus':  eta_dot_plus
            }
        }

    def learn(self, data):
        pass

    def f(self,t,x,u):
        return self.plant.f(t,x,u)

class Koopman(DynamicModel):
    def __init__(self, dynamic_plant: dfl.dynamic_system.DFLDynamicPlant, n_koop: int, dt_data: float=DT_DATA_DEFAULT, dt_control: float=DT_CTRL_DEFAULT, observable='polynomial'):
        if isinstance(observable, str):
            if observable == 'polynomial':
                self.g = lambda x : Koopman.g_koop_poly(x,n_koop)
            elif observable == 'fourier':
                self.g = lambda x : Koopman.g_koop_fourier(x,n_koop)
            elif observable == 'filippos':
                self.g = Koopman.gkoop2
            else:
                raise KeyError('Unsupported Observable: {}'.format(observable))
        else:
            self.g = observable

        super().__init__(dynamic_plant, dt_data, dt_control, name='Koopman')

    def learn(self, data: np.ndarray, dmd: bool=False):
        if dmd:
            x = np.concatenate((data['x']['data'],data['eta']['data']),2)
        else:
            x = data['x']['data']

        _, y_minus, y_plus = DynamicModel.lift_space(x, self.g)

        self.regress_K_matrix(data['u']['minus'], y_minus, y_plus)

        self.trained = True

    def regress_K_matrix(self, U_minus: np.ndarray, Y_minus: np.ndarray, Y_plus: np.ndarray):
        omega = np.concatenate((Y_minus.reshape(-1, Y_minus.shape[-1]),
                                U_minus.reshape(-1, U_minus.shape[-1])),axis=1).T
        
        Y = Y_plus.reshape(-1, Y_plus.shape[-1]).T

        G = np.linalg.lstsq(omega.T,Y.T,rcond=None)[0].T

        self.A_disc_koop = G[:,:Y_plus.shape[-1] ] 
        self.B_disc_koop = G[:, Y_plus.shape[-1]:]

    def f(self, t: float, x: np.ndarray, u: np.ndarray):
        self.check_for_training()

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        y_plus = np.dot(self.A_disc_koop,x) + np.dot(self.B_disc_koop, u)

        return y_plus

    @staticmethod
    def g_koop_poly(x: np.ndarray, m: int):
        # Assert that we are operating on a single state estimate, x
        assert len(np.shape(x))==1

        # Initialize output
        y = []

        # Initialize polynomial degree
        deg = 1

        # Repeat until y is full:
        while len(y)<m:
            # For all monomials of x of degree deg:
            for phi in itertools.combinations_with_replacement(x, deg):
                # Append the value of the monomial to the output
                y.append(np.prod(phi))
            deg+= 1

        # Trim excess monomials from output
        return y[:m]

    @staticmethod
    def g_koop_fourier(x: np.ndarray, m: int):
        # Assert that we are operating on a single state estimate, x
        assert len(np.shape(x))==1

        output = np.copy(x)
        if len(output)<m:
            # Normalize x?
            temp = tuple([np.arange(m+1-len(output))]*len(x))
            c = np.array(np.meshgrid(*temp)).T.reshape(-1,len(x))
            output = np.concatenate((output,np.cos(np.pi*np.dot(c,x))),0).tolist()
        return output[:m]

    @staticmethod
    def gkoop1(x: np.ndarray):
        m = 1.0
        k11 = 0.2
        k13 = 2.0
        b1  = 3.0
        
        def phi_c1(q):
            e = k11*q + k13*q**3
            return e

        def phi_r1(f):
            e = b1*np.sign(f)*f**2
            return e

        q,v = x[0], x[1]
        y = np.array([q,v,phi_c1(q), phi_r1(v)])
        return y

    @staticmethod
    def gkoop2(x: np.ndarray):
        q,v = x[0],x[1]

        y = np.array([q,v,q**2,q**3,q**4,q**5,q**6,q**7,
                      v**2,v**3,v**4,v**5,v**6,v**7,v**9,v**11,v**13,v**15,v**17,v**19,
                      v*q,v*q**2,v*q**3,v*q**4,v*q**5,
                      v**2*q,v**2*q**2,v**2*q**3,v**2*q**4,v**2
                      *q**5,
                      v**3*q,v**3*q**2,v**3*q**3,v**3*q**4,v**3*q**5])
        return y

    def simulate_system(self, x_0: np.ndarray, u_func: Callable, t_f: float):
        u_minus = np.zeros((self.plant.n_u,1))
        xi_0 = self.g(x_0)
        g_func = lambda t,x,u : self.g(x)

        t,u,xi,y = super().simulate_system(xi_0, u_minus, t_f, u_func, self.f, g_func, continuous=False)
        return t, u, xi, y

class DFL(DynamicModel):
    def __init__(self, dynamic_plant: dfl.dynamic_system.DFLDynamicPlant, dt_data: float=DT_DATA_DEFAULT, dt_control: float=DT_CTRL_DEFAULT, ac_filter: bool=False):
        self.ac_filter = ac_filter
        super().__init__(dynamic_plant, dt_data, dt_control, name='DFL')

    def learn(self, data: np.ndarray):
        data = copy.deepcopy(data)
        self.generate_DFL_disc_model(data['x']['minus'], data['eta']['minus'], data['u']['minus'], data['eta']['plus'], data['u']['plus'])
        self.trained = True

    def generate_DFL_disc_model(self, X_minus: np.ndarray, Eta_minus: np.ndarray, U_minus: np.ndarray, Eta_plus: np.ndarray, U_plus: np.ndarray):
        # Flatten input data into matrices
        X_minus   = DynamicModel.flatten_trajectory_data(  X_minus).T
        Eta_minus = DynamicModel.flatten_trajectory_data(Eta_minus).T
        U_minus   = DynamicModel.flatten_trajectory_data(  U_minus).T
        Eta_plus  = DynamicModel.flatten_trajectory_data(Eta_plus ).T
        U_plus    = DynamicModel.flatten_trajectory_data(  U_plus ).T

        if self.ac_filter:
            # Compute anticausal filter
            self.D = DFL.regress_D_matrix(U_minus, Eta_minus)

            # Update B_x = B_x + A_eta*D
            self.plant.B_cont_x+= np.matmul(self.plant.A_cont_eta, self.D)

            # Filter input from eta
            Eta_minus-= np.matmul(self.D, U_minus)
            Eta_plus -= np.matmul(self.D, U_plus )

        # Assemble data matrix
        xi = np.concatenate((X_minus, Eta_minus, U_minus),axis=0)
        
        # Regress H
        H_disc = np.linalg.lstsq(xi.T,Eta_plus.T,rcond=None)[0].T
        
        # Extract components of H
        H_disc_x   = H_disc[:,                               :self.plant.n_x                                ]
        H_disc_eta = H_disc[:,self.plant.n_x                 :self.plant.n_x+self.plant.n_eta               ]
        H_disc_u   = H_disc[:,self.plant.n_x+self.plant.n_eta:self.plant.n_x+self.plant.n_eta+self.plant.n_u]

        # Update H_u = H_u + H_eta*D
        if self.ac_filter:
            H_disc_u+= np.matmul(H_disc_eta, self.D)

        # Convert A and B from continuous-time to discrete-time
        (A_disc_x, B_disc_x  ,_,_,_) = scipy.signal.cont2discrete((self.plant.A_cont_x     , self.plant.B_cont_x, 
                                                                   np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                                   self.dt_data)
        (_       , A_disc_eta,_,_,_) = scipy.signal.cont2discrete((self.plant.A_cont_x     , self.plant.A_cont_eta, 
                                                                   np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                                   self.dt_data)

        # Assemble discrete-time A and B matrices
        self.A_disc_dfl = np.block([[A_disc_x  , A_disc_eta],
                                    [H_disc_x  , H_disc_eta]])
        self.B_disc_dfl = np.block([[B_disc_x],
                                    [H_disc_u]])

    def regress_D_matrix(U_minus: np.ndarray, Eta_minus: np.ndarray):
        # Copy data
        U_minus = np.copy(U_minus)
        Eta_minus = np.copy(Eta_minus)

        # Mean-zero
        U_minus-= np.mean(U_minus,1)
        Eta_minus-= np.mean(Eta_minus,1)[:,None]

        # Regress
        return np.linalg.lstsq(np.matmul(U_minus, U_minus.T).T, np.matmul(Eta_minus, U_minus.T).T, rcond=None)[0].T

    def f(self, t: float, x: np.ndarray, u: np.ndarray):
        self.check_for_training()

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        y_plus = np.dot(self.A_disc_dfl,x) + np.dot(self.B_disc_dfl, u)

        return y_plus

    def simulate_system(self, x_0: np.ndarray, u_func: Callable, t_f: float, continuous: bool=False):
        u_minus = np.zeros(self.plant.n_u)
        eta_0 = self.plant.phi(0.0, x_0, u_minus)
        if self.ac_filter:
            eta_0-= np.squeeze(np.matmul(self.D, u_minus))
        xi_0 = np.concatenate((x_0,eta_0))

        if continuous == True:
            raise NotImplementedError('Continuous DFL simulation no longer supported')
        else:
            t,u,xi,y = super().simulate_system(xi_0, u_minus, t_f, u_func, self.f, self.plant.g, continuous = False)
            
        return t, u, xi, y

class L3(DynamicModel):
    class AC_Filter(Enum):
        NONE = 0
        LINEAR = 1
        NONLINEAR = 2

    def __init__(self, dynamic_plant: dfl.dynamic_system.DFLDynamicPlant, n_eta: int, dt_data: float=DT_DATA_DEFAULT, dt_control: float=DT_CTRL_DEFAULT, ac_filter: str='none', model_fn: str='model', retrain: bool=True, hidden_units_per_layer: int=256, num_hidden_layers: int=1, ignore_zeta: bool=False):
        self.n_x = dynamic_plant.n_x
        self.n_z = 0 if ignore_zeta else dynamic_plant.n_eta
        self.n_e = n_eta
        self.n_u = dynamic_plant.n_u

        self.hidden_units_per_layer = hidden_units_per_layer
        self.num_hidden_layers = num_hidden_layers

        if ac_filter == 'none':
            self.ac_filter = L3.AC_Filter.NONE
        elif ac_filter == 'linear':
            self.ac_filter = L3.AC_Filter.LINEAR
        elif ac_filter == 'nonlinear':
            self.ac_filter = L3.AC_Filter.NONLINEAR
        else:
            raise KeyError('Unsupported AC filter: {}'.format(ac_filter))

        self.model_fn = model_fn
        self.retrain = retrain
        self.ignore_zeta = ignore_zeta

        super().__init__(dynamic_plant, dt_data, dt_control, name='L3')

    def step(self, x_batch: torch.Tensor, y_batch: torch.Tensor, model: torch.nn.Module, loss_fn: Callable):
        # Send data to GPU if applicable
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Parse input
        x_tm1    = x_batch[:,                 :self.n_x         ]
        zeta_tm1 = x_batch[:,self.n_x         :self.n_x+self.n_z]
        u_tm1    = x_batch[:,self.n_x+self.n_z:                 ]

        # Parse output
        x_t      = y_batch[:,                 :self.n_x         ]
        zeta_t   = y_batch[:,self.n_x         :self.n_x+self.n_z]
        u_t      = y_batch[:,self.n_x+self.n_z:                 ]

        # Filter zeta -> zeta*
        if self.ac_filter == L3.AC_Filter.LINEAR:
            zeta_tm1-= torch.matmul(u_tm1, model.D)
            zeta_t  -= torch.matmul(u_t  , model.D)
        elif self.ac_filter == L3.AC_Filter.NONLINEAR:
            nu_tm1   = model.g_u(u_tm1)
            nu_t     = model.g_u(u_t  )
            zeta_tm1 = zeta_tm1-nu_tm1 # Note: this is now zeta^*_{t-1}
            zeta_t   = zeta_t  -nu_t   # Note: this is now zeta^*_t
        
        # Compute "ground truth" eta_t using twin model
        xs_t  = torch.cat((x_t, zeta_t), 1)
        eta_t = model.g(xs_t)

        # Propagate x, zeta, and eta using model
        x_hat, zeta_hat, eta_hat = model(x_tm1, zeta_tm1, u_tm1)

        # Return
        return loss_fn(x_t, x_hat) + loss_fn(zeta_t, zeta_hat) + loss_fn(eta_t, eta_hat)

    def train_model(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, title: str=None):
        # Reshape x and y to be vector of tensors
        x = torch.transpose(x,0,1)
        y = torch.transpose(y,0,1)

        # Split dataset into training and validation sets
        N_train = int(3*len(y)/5)
        dataset = torch.utils.data.TensorDataset(x, y)
        train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [N_train,len(y)-N_train])

        # Construct dataloaders for batch processing
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32)
        val_loader   = torch.utils.data.DataLoader(dataset=val_dataset  , batch_size=32)

        # Define learning hyperparameters
        loss_fn       = torch.nn.MSELoss(reduction='sum')
        learning_rate = .00001
        n_epochs      = 100000
        optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize arrays for logging
        training_losses = []
        validation_losses = []

        # Main training loop
        try:
            for t in range(n_epochs):
                # Validation
                with torch.no_grad():
                    losses = []
                    for x, y in val_loader:
                        loss = self.step(x, y, model, loss_fn)
                        losses.append(loss.item())
                    validation_losses.append(np.mean(losses))

                # Terminating condition
                if t>50 and np.mean(validation_losses[-20:-11])<=np.mean(validation_losses[-10:-1]):
                    break

                # Training
                losses = []
                for x, y in train_loader:
                    loss = self.step(x, y, model, loss_fn)
                    losses.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                training_losses.append(np.mean(losses))

                pstr = f"[{t+1}] Training loss: {training_losses[-1]:.3f}\t Validation loss: {validation_losses[-1]:.3f}"

                print(pstr)
        except KeyboardInterrupt:
            print('Stopping due to keyboard interrupt. Save and continue (Y/N)?')
            ans = input()
            if ans[0].upper() == 'N':
                exit(0)

        fig, axs = plt.subplots(1,1)
        axs.semilogy(range(len(  training_losses)),   training_losses, label=  'Training Loss')
        axs.semilogy(range(len(validation_losses)), validation_losses, label='Validation Loss')
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Loss')
        axs.legend()
        if title is not None:
            axs.title(title)

        model.eval()
        return model

    def learn(self, data: dict):
        # Copy data for manipulation
        data = copy.deepcopy(data)

        # Format data
        x_minus = torch.transpose(torch.from_numpy(DynamicModel.flatten_trajectory_data(data['x'  ]['minus'])).type(dtype), 0,1)
        z_minus = torch.transpose(torch.from_numpy(DynamicModel.flatten_trajectory_data(data['eta']['minus'])).type(dtype), 0,1)
        u_minus = torch.transpose(torch.from_numpy(DynamicModel.flatten_trajectory_data(data['u'  ]['minus'])).type(dtype), 0,1)
        x_plus  = torch.transpose(torch.from_numpy(DynamicModel.flatten_trajectory_data(data['x'  ]['plus' ])).type(dtype), 0,1)
        z_plus  = torch.transpose(torch.from_numpy(DynamicModel.flatten_trajectory_data(data['eta']['plus' ])).type(dtype), 0,1)
        u_plus  = torch.transpose(torch.from_numpy(DynamicModel.flatten_trajectory_data(data['u'  ]['plus' ])).type(dtype), 0,1)

        # Initialize model
        if self.ac_filter == L3.AC_Filter.NONLINEAR:
            self.model = L3Module.ILDFL     (self.n_x, self.n_z, self.n_e, self.n_u, self.hidden_units_per_layer)
        else:
            self.model = L3Module.LearnedDFL(self.n_x, self.n_z, self.n_e, self.n_u, self.hidden_units_per_layer, hidden_layers=self.num_hidden_layers)

        # If anticausal filter is linear, learn D
        if self.ac_filter == L3.AC_Filter.LINEAR:
            self.model.regress_D_matrix(torch.transpose(u_minus, 0,1), torch.transpose(z_minus, 0,1))

        # Train/load model
        if self.retrain:
            if self.ignore_zeta:
                self.model = self.train_model(self.model, torch.cat((x_minus, u_minus), 0), torch.cat((x_plus,u_plus), 0))
            else:
                self.model = self.train_model(self.model, torch.cat((x_minus, z_minus, u_minus), 0), torch.cat((x_plus,z_plus,u_plus), 0))
            torch.save(self.model.state_dict(), '{}.pt'.format(self.model_fn))
        else:
            self.model.load_state_dict(torch.load('{}.pt'.format(self.model_fn)))
        self.model.filter_linear_model()

        def augmented_state(x, z, u):
            x_shape = x.shape
            if len(x_shape)==3:
                x = x.reshape(-1, x.shape[-1])
                z = z.reshape(-1, z.shape[-1])
                u = u.reshape(-1, u.shape[-1])

            x = torch.from_numpy(x.T).type(dtype)
            z = torch.from_numpy(z.T).type(dtype)
            u = torch.from_numpy(u.T).type(dtype)

            if self.ignore_zeta:
                xs = x
            else:
                if self.ac_filter == L3.AC_Filter.LINEAR:
                    z -= torch.matmul(u, self.model.D)
                elif self.ac_filter == L3.AC_Filter.NONLINEAR:
                    nu = self.model.g_u(u)
                    z -= self.model.D(nu)
                xs = torch.cat((x,z), 0)
            eta = self.model.g(xs)

            xs = torch.cat((xs,eta), 0)

            return xs.detach().numpy()

        self.augmented_state = augmented_state

        self.trained = True

    def regress_new_LDM(self, data):
        # Copy data for manipulation
        data = copy.deepcopy(data)

        # Format data
        x_minus = DynamicModel.flatten_trajectory_data(data['x'  ]['minus'])
        z_minus = DynamicModel.flatten_trajectory_data(data['eta']['minus'])
        u_minus = DynamicModel.flatten_trajectory_data(data['u'  ]['minus'])
        x_plus  = DynamicModel.flatten_trajectory_data(data['x'  ]['plus' ])
        z_plus  = DynamicModel.flatten_trajectory_data(data['eta']['plus' ])
        u_plus  = DynamicModel.flatten_trajectory_data(data['u'  ]['plus' ])

        xs_minus = []
        xs_plus = []
        for i in range(len(x_minus)):
            xa_minus = self.augmented_state(x_minus[i], z_minus[i], u_minus[i])
            xa_plus  = self.augmented_state(x_plus [i], z_plus [i], u_plus [i])

            xs_minus.append(np.concatenate((xa_minus, u_minus[i]),0))
            xs_plus.append(xa_plus)

        xs_minus = np.asarray(xs_minus)
        xs_plus  = np.asarray(xs_plus )

        ldm = np.linalg.lstsq(xs_minus,xs_plus,rcond=None)[0].T

        self.model.A.weight.data = torch.from_numpy(ldm[                 :self.n_x         ,:]).type(dtype)
        self.model.Z.weight.data = torch.from_numpy(ldm[self.n_x         :self.n_x+self.n_z,:]).type(dtype)
        self.model.H.weight.data = torch.from_numpy(ldm[self.n_x+self.n_z:                 ,:]).type(dtype)

    def f(self, t: float, xs: np.ndarray, u:np.ndarray):
        self.check_for_training()

        if not isinstance(u, np.ndarray):
            u = np.array([u])

        # Parse input
        x_t  = xs[                 :self.n_x         ]
        zs_t = xs[self.n_x         :self.n_x+self.n_z]
        e_t  = xs[self.n_x+self.n_z:                 ]

        # NumPy to PyTorch
        x_t  = torch.from_numpy( x_t).type(dtype)
        zs_t = torch.from_numpy(zs_t).type(dtype)
        e_t  = torch.from_numpy( e_t).type(dtype)
        u    = torch.from_numpy( u  ).type(dtype)

        # Assemble into xi
        xi_t = torch.cat((x_t,zs_t,e_t,u), 0)

        # Propagate through model
        x_tp1, zs_tp1, eta_tp1 = self.model.ldm(xi_t)

        # PyTorch to NumPy
        x_tp1   =   x_tp1.detach().numpy()
        zs_tp1  =  zs_tp1.detach().numpy()
        eta_tp1 = eta_tp1.detach().numpy()

        return np.concatenate((x_tp1,zs_tp1,eta_tp1))

    def simulate_system(self, xs_0: np.ndarray, u_func: Callable, t_f: float):
        x_0 = xs_0[:self.n_x]
        u_0 = np.zeros(self.n_u)
        if self.ignore_zeta:
            z_0 = np.array([])
        else:
            z_0 = xs_0[self.n_x:] if len(xs_0)>self.n_x else self.plant.phi(0,x_0,u_0)
        assert len(z_0)==self.n_z

        xs_0 = self.augmented_state(x_0, z_0, u_0)

        t,u,xi,y = super().simulate_system(xs_0, u_0, t_f, u_func, self.f, self.plant.g, continuous=False)

        return t, u, xi, y