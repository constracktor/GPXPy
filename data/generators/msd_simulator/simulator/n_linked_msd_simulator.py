import os
import json
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp
import matplotlib
import matplotlib.pyplot as plt

class n_linked_msd_simulator():
    def __init__(self, config_filepath):
        # Open config
        config = self.get_config(config_filepath)        # Number of samples to generate
        self.n_samples = config["N_SAMPLES"]
        # Sampling frequency
        self.f_sample = config["SAMPLING_FREQUENCY"]
        # Measurement noise
        self.noise_sd = config["NOISE"]
        # Number of carts
        self.N = config["N_CARTS"]

        self.signal = config["SIGNAL"]
        self.amplitude = config["AMPLITUDE"] # aprbs=2.0; sinus=4.0
        self.period = config["PERIOD"]       # aprbs=300; sinus=5
        
        # Model parameters
        self.k = np.linspace(1.0, 0.5, self.N)
        self.c = 0.5 * np.linspace(0.5, 1.0, self.N)
        self.m = 0.5 * np.linspace(0.5, 1, self.N)
        # Spring travel
        self.travel = 2
        # Random seed for reproducibility
        np.random.seed(config["RANDOM_SEED"])

    # For displacements x, returns the spring force
    def spring_func(self, x):
        # piecewise linear spring function
        kf = 0.25
        d = self.travel / 2
        f = kf * x * (x < d) * (x > -d)
        f = f + (x - d + kf * d) * (x >= d)
        f = f + (x + d - kf * d) * (x <= -d)
        return f

    def dynamics(self, x, u):
        # Model parameters
        k = self.k
        c = self.c
        m = self.m
        # Separate state vector x into positions and velocities
        d = x[0::2, :]
        v = x[1::2, :]
        # Vector containing the forces on each cart
        F = 0 * (d)
        # Force on first cart
        F[0] = F[0] + (u + k[0] * self.spring_func(-d[0])
                       + k[1] * self.spring_func(d[1] - d[0])
                       - c[0] * v[0] + c[1] * (v[1] - v[0]))
        # Force on the middle carts
        F[1:-1] = F[1:-1] + (k[1:-1, None] * self.spring_func(d[0:-2, :] - d[1:-1, :]) +
                             k[2:, None] * self.spring_func(d[2:, :] - d[1:-1, :]) +
                             c[1:-1, None] * (v[0:-2, :] - v[1:-1, :]) +
                             c[2:, None] * (v[2:, :] - v[1:-1, :]))
        # Force on the last cart
        F[-1] = F[-1] + (k[-1, None] * self.spring_func(d[-2, :] - d[-1, :])
                         + c[-1, None] * (v[-2, :] - v[-1, :]))
        # Fill derivative with velocities and accelerations
        dxdt = 0 * x
        dxdt[0::2] = v
        dxdt[1::2] = F / m[:, None]
        return dxdt

    def simulate(self, u):
        T = u.size
        Tend = self.f_sample * T
        time = np.linspace(0, Tend, T)
        u_interp = interp.interp1d(time, u)

        # Construct function for the dynamcis of the system
        x0 = np.zeros((2 * self.N))
        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t)[None])
            dx = dX.reshape(-1)
            return dx
        # Solve the initial value problem with a
        sol = integrate.solve_ivp(dyn, [0.0, Tend], x0, t_eval=time, method='RK45')
        Y = sol['y'][-2:-1][0]
        # Add noise
        Y = Y + (np.random.normal(0, self.noise_sd, Y.shape))
        return Y
    
    #  Generate a piecewise constant input with randomly varying period and magnitude
    def aprbs_signal(self, T, amplitude=1, period=50):
        u_per = []
        while(sum(u_per) < T):
            u_per += [int((period * np.random.rand()))]

        # shorten the last element so that the periods add up to the total length
        u_per[-1] = u_per[-1] - (sum(u_per) - T)

        u = np.concatenate([amplitude * (np.random.rand() - 0.5) * np.ones((per, 1)) for per in u_per])
        return u.reshape(1,-1)[0]

    # Generate a multisine input with randomly varying period and amplitude
    def multisine_signal(self, T, amplitude=2, period=10):
        samples = np.linspace(0,T,T)
        n_sinus = 40
        amplitudes = amplitude * np.random.rand(n_sinus)
        periods = period * np.random.rand(n_sinus) / 1000
        offset = np.random.rand(n_sinus) - 0.5
        signal = amplitudes[0] * np.sin(np.pi * periods[0] * samples + offset[0])
        for i in range(1,n_sinus):
            signal = signal + amplitudes[i] * np.sin(np.pi * periods[i] * samples + offset[i])
        signal = signal / n_sinus
        return signal

    def generate_input(self):
        if (self.signal == "aprbs"):
            return self.aprbs_signal(self.n_samples, self.amplitude, self.period)
        else: # config["SIGNAL"] == "sinus"
            return self.multisine_signal(self.n_samples, self.amplitude, self.period)
    
    def write_data(self, filepath, data):
        # Ensure directory exists
        os.makedirs(filepath[:filepath.rfind("/")], exist_ok=True)
        file = open(filepath, "w")
        for value in data:
            file.write(str(value) + "\n")
        file.close()

    def plot_data(self, data, label):
        time = np.linspace(0, self.f_sample*self.n_samples, self.n_samples)
        plt.figure(figsize=(10,4))
        plt.plot(time, data, 'k', linewidth=1)
        plt.ylabel("Position")
        plt.xlabel("Time")
        plt.title(label)
        # Ensure directory exists
        os.makedirs("plots", exist_ok=True)
        plt.savefig('./plots/' + label + '.pdf', bbox_inches='tight')

    # Rescale data between [a, b]
    def feature_scale_data(self, data, a=0.0, b=1.0):
        return a + (data - np.min(data)) * (b - a) / (np.max(data) - np.min(data))

    def get_config(self, filepath):
        return json.load(open(filepath))
