# Python Packages
import numpy as np
import sympy as sp
import random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Simulation settings
dt = 0.01   # Simulation time step - sec
t_final = 5   # Simulation run time - sec
steps = round(t_final/dt)   #Number of simulation ssteps
t = np.zeros(steps+1)   # Initializes time vector

# Noise
Q_t = 0.001*np.identity(9)   # Process noise
R_t = 0.001*np.identity(3)   # Measurement noise

# Initialization
# System
m = 3
n = 9

#Parameters
m_Q = 5
m_P = 2
I_yy = 0.01
grav = 9.81
l = 1

#Dynamics
state_initial = np.zeros(n)
state_initial[n-1] = m_P  # Starts at origin with flat orientation and mass hanging straight down

# List of states-
# 0 - x
# 1 - z
# 2 - theta
# 3 - phi
# 4 - xdot
# 5 - zdot
# 6 - thetadot
# 7 - phidot
# 8 - mp

x = sp.symbols('x')
z = sp.symbols('z')
theta = sp.symbols('theta')
phi = sp.symbols('phi')
x_dot = sp.symbols('x_dot')
z_dot = sp.symbols('z_dot')
theta_dot = sp.symbols('theta_dot')
phi_dot = sp.symbols('phi_dot')
m_p = sp.symbols('m_p')

u1 = sp.symbols('u1')
u2 = sp.symbols('u2')

# Control
thrust_initial = (m_Q + m_P)*grav  # The initial control output is currently set to a hovering thrust
tau_initial = 0
control_initial = np.zeros(2)
control_initial[0] = thrust_initial
control_initial[1] = tau_initial

# EKF Initialization
mu_initial = np.zeros(n)   # Mean of initial guess for state
mu_initial[n-1] = 1   # Guesses that the quadcopter starts stationary at the origin and has a payload of mass 1 kg
Sigma_initial = 0.1*np.identity(9)   # Covariance of initial guess for state

def Simulation(steps, dt, state_initial, control_initial, mu_initial, Sigma_initial, loaded, m_p_var):

    [A, C] = Jacob(x, z, theta, phi, x_dot, z_dot, theta_dot, phi_dot, u1, u2)

    state = np.zeros((steps + 1, n))
    state[0, :] = state_initial

    control = np.zeros((steps + 1, 2))
    control[0, :] = control_initial

    y = np.zeros((steps + 1, 3))  # Measurement vector
    y[0, :] = np.NaN  # Sets measurements at t=0 to not plot since they do not exist

    distribution = multivariate_normal(mu_initial, Sigma_initial)  # Define distribution from which to sample initial guess for EKF

    mu_t_t = np.zeros((steps + 1, n))  # Mean update vector
    mu_t_t[0, :] = distribution.rvs()  # Initial guess for state

    Sigma_t_t = np.zeros((steps + 1, n, n))  # Prediction update vector
    Sigma_t_t[0, :, :] = Sigma_initial  # Initial guess for covariance

    upper_conf_int = np.zeros((steps + 1, n))  # Vector for upper confidence interval
    lower_conf_int = np.zeros((steps + 1, n))  # Vector for lower confidence interval

    for j in range(n):
        upper_conf_int[0, j] = mu_t_t[0, j] + 1.96 * np.sqrt(Sigma_t_t[0, j, j])
        lower_conf_int[0, j] = mu_t_t[0, j] - 1.96 * np.sqrt(Sigma_t_t[0, j, j])

    for i in range(steps):

        t[i + 1] = dt * i  # Moves simulation forward by one time step

        # Control
        control = Control(state, control, i)

        # Dynamics

        if loaded:
            state = Dynamics_Loaded(state, control, i)
        else:
            state = Dynamics_Unloaded(state, control, i)

        # Measurement
        y = Measure(state, y, i)

        # Jacobians
        A_t = A(mu_t_t[i, 0], mu_t_t[i, 1], mu_t_t[i, 2], mu_t_t[i, 3], mu_t_t[i, 4], mu_t_t[i, 5], mu_t_t[i, 6], mu_t_t[i, 7], mu_t_t[i, 8], control[i, 0], control[i, 1])
        A_t = np.array(A_t).astype(np.float64)
        C_t = np.array(C).astype(np.float64)

        # EKF
        if loaded:
            [mu_t_plus_t, Sigma_t_plus_t] = EKF_Predict_Loaded(mu_t_t, Sigma_t_t, control, A_t, i)
        else:
            [mu_t_plus_t, Sigma_t_plus_t] = EKF_Predict_Unloaded(mu_t_t, Sigma_t_t, control, A_t, i)

        [mu_t_t, Sigma_t_t] = EKF_Update(mu_t_t, Sigma_t_t, mu_t_plus_t, Sigma_t_plus_t, C_t, y, i)

        # Confidence Intervals

        [upper_conf_int, lower_conf_int] = Confidence(upper_conf_int, lower_conf_int, mu_t_t, Sigma_t_t, i)

    return state, control, y, mu_t_t, Sigma_t_t, upper_conf_int, lower_conf_int

def Control(state, control, i):

    control[i + 1, 0] = (m_Q + m_P) * grav + i*dt*10
    control[i + 1, 1] = 0

    return control

def Dynamics_Loaded(state, control, i):

    state[i + 1, 0] = state[i, 0] + dt * state[i, 4] + random.gauss(0, Q_t[0, 0])
    state[i + 1, 1] = state[i, 1] + dt * state[i, 5] + random.gauss(0, Q_t[1, 1])
    state[i + 1, 2] = state[i, 2] + dt * state[i, 6] + random.gauss(0, Q_t[2, 2])
    state[i + 1, 3] = state[i, 3] + dt * state[i, 7] + random.gauss(0, Q_t[3, 3])
    state[i + 1, 4] = state[i, 4] + dt * control[i, 0] * np.sin(state[i, 2]) * (m_Q + state[i, 8] * (np.cos(state[i, 3])) ** 2) / (m_Q * (m_Q + state[i, 8])) + dt * control[i, 0] * np.cos(state[i, 2]) * (state[i, 8] * np.sin(state[i, 3]) * np.cos(state[i, 3])) / (m_Q * (m_Q + state[i, 8])) + dt * state[i, 8] * l * (state[i, 7]) ** 2 * np.sin(state[i, 3]) / (m_Q + state[i, 8]) + random.gauss(0, Q_t[4, 4])
    state[i + 1, 5] = state[i, 5] + dt * control[i, 0] * np.cos(state[i, 2]) * (m_Q + state[i, 8] * (np.sin(state[i, 3])) ** 2) / (m_Q * (m_Q + state[i, 8])) + dt * control[i, 0] * np.sin(state[i, 2]) * (state[i, 8] * np.sin(state[i, 3]) * np.cos(state[i, 3])) / (m_Q * (m_Q + state[i, 8])) - dt * state[i, 8] * l * (state[i, 7]) ** 2 * np.sin(state[i, 3]) / (m_Q + state[i, 8]) - dt * grav + random.gauss(0, Q_t[5, 5])
    state[i + 1, 6] = state[i, 6] + dt * control[i, 1] / I_yy + random.gauss(0, Q_t[6, 6])
    state[i + 1, 7] = state[i, 7] - dt * control[i, 0] * np.sin(state[i, 3] - state[i, 2]) / (m_Q * l) + random.gauss(0,Q_t[7, 7])
    state[i + 1, 8] = state[i, 8]

    return state

def Dynamics_Unloaded(state, control, i):

    state[i + 1, 0] = state[i, 0] + dt * state[i, 4] + random.gauss(0, Q_t[0, 0])
    state[i + 1, 1] = state[i, 1] + dt * state[i, 5] + random.gauss(0, Q_t[1, 1])
    state[i + 1, 2] = state[i, 2] + dt * state[i, 6] + random.gauss(0, Q_t[2, 2])
    state[i + 1, 3] = state[i, 3] + dt * state[i, 7]
    state[i + 1, 4] = state[i, 4] + dt * control[i, 0] * np.sin(state[i, 2]) / m_Q
    state[i + 1, 5] = state[i, 5] + dt * control[i, 0] * np.cos(state[i, 2]) / m_Q - dt * grav + random.gauss(0, Q_t[5, 5])
    state[i + 1, 6] = state[i, 6] + dt * control[i, 1] / I_yy + random.gauss(0, Q_t[6, 6])
    state[i + 1, 7] = 0
    state[i + 1, 8] = state[i, 8]

    return state

def Dynamics_No_m_p(state, control, i):
    state[i + 1, 0] = state[i, 0] + dt * state[i, 4] + random.gauss(0, Q_t[0, 0])
    state[i + 1, 1] = state[i, 1] + dt * state[i, 5] + random.gauss(0, Q_t[1, 1])
    state[i + 1, 2] = state[i, 2] + dt * state[i, 6] + random.gauss(0, Q_t[2, 2])
    state[i + 1, 3] = state[i, 3] + dt * state[i, 7] + random.gauss(0, Q_t[3, 3])
    state[i + 1, 4] = state[i, 4] + dt * control[i, 0] * np.sin(state[i, 2]) * (m_Q + m_P * (np.cos(state[i, 3])) ** 2) / (m_Q * (m_Q + m_P)) + dt * control[i, 0] * np.cos(state[i, 2]) * (m_P * np.sin(state[i, 3]) * np.cos(state[i, 3])) / (m_Q * (m_Q + m_P)) + dt * m_P * l * (state[i, 7]) ** 2 * np.sin(state[i, 3]) / (m_Q + m_P) + random.gauss(0, Q_t[4, 4])
    state[i + 1, 5] = state[i, 5] + dt * control[i, 0] * np.cos(state[i, 2]) * (m_Q + m_P * (np.sin(state[i, 3])) ** 2) / (m_Q * (m_Q + m_P)) + dt * control[i, 0] * np.sin(state[i, 2]) * (m_P * np.sin(state[i, 3]) * np.cos(state[i, 3])) / (m_Q * (m_Q + m_P)) - dt * m_P * l * (state[i, 7]) ** 2 * np.sin(state[i, 3]) / (m_Q + m_P) - dt * grav + random.gauss(0, Q_t[5, 5])
    state[i + 1, 6] = state[i, 6] + dt * control[i, 1] / I_yy + random.gauss(0, Q_t[6, 6])
    state[i + 1, 7] = state[i, 7] - dt * control[i, 0] * np.sin(state[i, 3] - state[i, 2]) / (m_Q * l) + random.gauss(0, Q_t[7, 7])

    return state

def Measure(state, y, i):

    y[i + 1, 0] = state[i + 1, 0] + random.gauss(0, R_t[0, 0])
    y[i + 1, 1] = state[i + 1, 1] + random.gauss(0, R_t[1, 1])
    y[i + 1, 2] = state[i + 1, 2] + random.gauss(0, R_t[2, 2])

    return y

def EKF_Predict_Loaded(mu_t_t, Sigma_t_t, control, A_t, i):

    f = np.zeros(n)

    f[0] = mu_t_t[i, 0] + dt * mu_t_t[i, 4]
    f[1] = mu_t_t[i, 1] + dt * mu_t_t[i, 5]
    f[2] = mu_t_t[i, 2] + dt * mu_t_t[i, 6]
    f[3] = mu_t_t[i, 3] + dt * mu_t_t[i, 7]
    f[4] = mu_t_t[i, 4] + dt * control[i, 0] * np.sin(mu_t_t[i, 2]) * (m_Q + mu_t_t[i, 8] * (np.cos(mu_t_t[i, 3])) ** 2) / (m_Q * (m_Q + mu_t_t[i, 8])) + dt * control[i, 0] * np.cos(mu_t_t[i, 2]) * (mu_t_t[i, 8] * np.sin(mu_t_t[i, 3]) * np.cos(mu_t_t[i, 3])) / (m_Q * (m_Q + mu_t_t[i, 8])) + dt * mu_t_t[i, 8] * l * (mu_t_t[i, 7]) ** 2 * np.sin(mu_t_t[i, 3]) / (m_Q + mu_t_t[i, 8])
    f[5] = mu_t_t[i, 5] + dt * control[i, 0] * np.cos(mu_t_t[i, 2]) * (m_Q + mu_t_t[i, 8] * (np.sin(mu_t_t[i, 3])) ** 2) / (m_Q * (m_Q + mu_t_t[i, 8])) + dt * control[i, 0] * np.sin(mu_t_t[i, 2]) * (mu_t_t[i, 8] * np.sin(mu_t_t[i, 3]) * np.cos(mu_t_t[i, 3])) / (m_Q * (m_Q + mu_t_t[i, 8])) - dt * mu_t_t[i, 8] * l * (mu_t_t[i, 7]) ** 2 * np.sin(mu_t_t[i, 3]) / (m_Q + mu_t_t[i, 8]) - dt * grav
    f[6] = mu_t_t[i, 6] + dt * control[i, 1] / I_yy
    f[7] = mu_t_t[i, 7] - dt * control[i, 0] * np.sin(mu_t_t[i, 3] - mu_t_t[i, 2]) / (m_Q * l)
    f[8] = mu_t_t[i, 8]

    mu_t_plus_t = f
    Sigma_t_plus_t = A_t @ Sigma_t_t[i, :, :] @ A_t.T + Q_t

    return mu_t_plus_t, Sigma_t_plus_t

def EKF_Predict_Unloaded(mu_t_t, Sigma_t_t, control, A_t, i):

    f = np.zeros(n)

    f[0] = mu_t_t[i, 0] + dt * mu_t_t[i, 4]
    f[1] = mu_t_t[i, 1] + dt * mu_t_t[i, 5]
    f[2] = mu_t_t[i, 2] + dt * mu_t_t[i, 6]
    f[3] = mu_t_t[i, 3] + dt * mu_t_t[i, 7]
    f[4] = mu_t_t[i, 4] + dt * control[i, 0] * np.sin(mu_t_t[i, 2]) / m_Q
    f[5] = mu_t_t[i, 5] + dt * control[i, 0] * np.sin(mu_t_t[i, 2]) / m_Q - dt * grav
    f[6] = mu_t_t[i, 6] + dt * control[i, 1] / I_yy
    f[7] = mu_t_t[i, 7]
    f[8] = mu_t_t[i, 8]

    mu_t_plus_t = f
    Sigma_t_plus_t = A_t @ Sigma_t_t[i, :, :] @ A_t.T + Q_t

    return mu_t_plus_t, Sigma_t_plus_t

def EKF_Update(mu_t_t, Sigma_t_t, mu_t_plus_t, Sigma_t_plus_t, C_t, y, i):

    g = np.zeros(m)

    g[0] = mu_t_t[i, 0]
    g[1] = mu_t_t[i, 1]
    g[2] = mu_t_t[i, 2]

    mu_t_t[i + 1, :] = mu_t_plus_t + Sigma_t_plus_t @ C_t.T @ np.linalg.inv(C_t @ Sigma_t_plus_t @ C_t.T + R_t) @ (y[i + 1, :] - g)
    Sigma_t_t[i + 1, :, :] = Sigma_t_plus_t - Sigma_t_plus_t @ C_t.T @ np.linalg.inv(C_t @ Sigma_t_plus_t @ C_t.T + R_t) @ C_t @ Sigma_t_plus_t

    return mu_t_t, Sigma_t_t

def Confidence(upper_conf_int, lower_conf_int, mu_t_t, Sigma_t_t, i):

    for j in range(n):
        upper_conf_int[i + 1, j] = mu_t_t[i + 1, j] + 1.96 * np.sqrt(Sigma_t_t[i + 1, j, j])
        lower_conf_int[i + 1, j] = mu_t_t[i + 1, j] - 1.96 * np.sqrt(Sigma_t_t[i + 1, j, j])

    return upper_conf_int, lower_conf_int

def Jacob(x, z, theta, phi, x_dot, z_dot, theta_dot, phi_dot, u1, u2):

    eqn1 = x + dt * x_dot
    eqn2 = z + dt * z_dot
    eqn3 = theta + dt * theta_dot
    eqn4 = phi + dt * phi_dot
    eqn5 = x_dot + dt * ((m_Q + m_p * (sp.cos(phi)) ** 2) / (m_Q * (m_Q + m_p)) * u1 * sp.sin(theta) + (
                m_p * sp.sin(phi) * sp.cos(phi)) / (m_Q * (m_Q + m_p)) * u1 * sp.cos(theta) + (
                                     m_p * l * phi_dot ** 2 * sp.sin(phi)) / (m_Q + m_p))
    eqn6 = z_dot + dt * ((m_Q + m_p * (sp.sin(phi)) ** 2) / (m_Q * (m_Q + m_p)) * u1 * sp.cos(theta) + (
                m_p * sp.sin(phi) * sp.cos(phi)) / (m_Q * (m_Q + m_p)) * u1 * sp.sin(theta) - (
                                     m_p * l * phi_dot ** 2 * sp.cos(phi)) / (m_Q + m_p) - grav)
    eqn7 = theta_dot + dt * u2 / I_yy
    eqn8 = phi_dot - dt * u1 * sp.sin(phi - theta) / (m_Q * l)
    eqn9 = m_p

    F = sp.Matrix([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9])
    G = sp.Matrix([x, z, theta])
    X = sp.Matrix([x, z, theta, phi, x_dot, z_dot, theta_dot, phi_dot, m_p])

    A = sp.simplify(F.jacobian(X))
    C = sp.simplify(G.jacobian(X))

    A = sp.lambdify([x, z, theta, phi, x_dot, z_dot, theta_dot, phi_dot, m_p, u1, u2], A)

    return A, C


# Simulation loop

[state, control, y, mu_t_t, Sigma_t_t, upper_conf_int, lower_conf_int] = Simulation(steps, dt, state_initial, control_initial, mu_initial, Sigma_initial, 1, 1)



# Plots

plt.figure(1)
plt.plot(t, state[:, 1], label='True')
plt.plot(t, mu_t_t[:, 1], label='Belief')
plt.fill_between(t, upper_conf_int[:, 1], lower_conf_int[:, 1], color='green', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('Time (sec)')
plt.ylabel('Position in z (m)')
plt.legend()
plt.show()