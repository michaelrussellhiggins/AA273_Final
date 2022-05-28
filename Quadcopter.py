# Python Packages
import numpy as np
import sympy as sp
import random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Simulation settings
dt = 0.01   # Simulation time step - sec
t_final = 5   # Simulation run time - sec
steps = round(t_final/dt)
t = np.zeros(steps+1)   # Initializes time vector

# Noise
Q_t = 0.001*np.identity(9)   # Process noise
R_t = 0.001*np.identity(3)   # Measurement noise

# Initialization
# Dynamics
m = 3
n = 9

m_Q = 5
m_P = 2
I_yy = 0.01
grav = 9.81
l = 1

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

# EKF Initialization
mu_initial = np.zeros(n)   # Mean of initial guess for state
mu_initial[n-1] = 1   # Guesses that the quadcopter starts stationary at the origin and has a payload of mass 1 kg
Sigma_initial = 0.1*np.identity(9)   # Covariance of initial guess for state

def Simulation(steps, dt, state_initial, thrust_initial, tau_initial, mu_initial, Sigma_initial):

    [A, C] = Jacob(x, z, theta, phi, x_dot, z_dot, theta_dot, phi_dot, u1, u2)

    state = np.zeros((n, steps + 1))
    state[:, 0] = state_initial

    thrust = np.zeros(steps + 1)  # Thrust vector
    tau = np.zeros(steps + 1)  # Torque vector
    thrust[0] = thrust_initial  # Initial thrust
    tau[0] = tau_initial  # Initial torque

    y = np.zeros((3, steps + 1))  # Measurement vector
    y[:, 0] = np.NaN  # Sets measurements at t=0 to not plot since they do not exist

    distribution = multivariate_normal(mu_initial,Sigma_initial)  # Define distribution from which to sample initial guess for EKF

    mu_t_t = np.zeros((n, steps + 1))  # Mean update vector
    mu_t_t[:, 0] = distribution.rvs()  # Initial guess for state

    Sigma_t_t = np.zeros((n, n, steps + 1))  # Prediction update vector
    Sigma_t_t[:, :, 0] = Sigma_initial  # Initial guess for covariance

    upper_conf_int = np.zeros((n, steps + 1))  # Vector for upper confidence interval
    lower_conf_int = np.zeros((n, steps + 1))  # Vector for lower confidence interval

    for j in range(n):
        upper_conf_int[j, 0] = mu_t_t[j, 0] + 1.96 * np.sqrt(Sigma_t_t[j, j, 0])
        lower_conf_int[j, 0] = mu_t_t[j, 0] - 1.96 * np.sqrt(Sigma_t_t[j, j, 0])

    for i in range(steps):

        t[i + 1] = dt * i  # Moves simulation forward by one time step

        # Control
        [thrust, tau] = Control(state, thrust, tau, i)

        # Dynamics
        state = Dynamics(state, thrust, tau, i)

        # Measurement
        y = Measure(state, y, i)

        # Jacobians
        A_mid = A.subs([(x, mu_t_t[0,i]), (z, mu_t_t[1,i]), (theta, mu_t_t[2,i]), (phi, mu_t_t[3,i]), (x_dot, mu_t_t[4,i]), (z_dot, mu_t_t[5,i]), (theta_dot, mu_t_t[6,i]), (phi_dot, mu_t_t[7,i]), (m_p, mu_t_t[8,i]), (u1, thrust[i]), (u2, tau[i])])
        A_t = np.array(A_mid).astype(np.float64)
        C_t = np.array(C).astype(np.float64)

        # EKF
        [mu_t_plus_t, Sigma_t_plus_t] = EKF_Predict(mu_t_t, Sigma_t_t, thrust, tau, A_t, i)
        [mu_t_t, Sigma_t_t] = EKF_Update(mu_t_t, Sigma_t_t, mu_t_plus_t, Sigma_t_plus_t, C_t, y, i)

        # Confidence Intervals

        [upper_conf_int, lower_conf_int] = Confidence(upper_conf_int, lower_conf_int, mu_t_t, Sigma_t_t, i)

    return state, thrust, tau, y, mu_t_t, Sigma_t_t, upper_conf_int, lower_conf_int

def Control(state, thrust, tau, i):

    thrust[i + 1] = (m_Q + m_P) * grav + i*dt*10
    tau[i + 1] = 0

    return thrust, tau

def Dynamics(state, thrust, tau, i):

    state[0, i + 1] = state[0, i] + dt * state[4, i] + random.gauss(0, Q_t[0, 0])
    state[1, i + 1] = state[1, i] + dt * state[5, i] + random.gauss(0, Q_t[1, 1])
    state[2, i + 1] = state[2, i] + dt * state[6, i] + random.gauss(0, Q_t[2, 2])
    state[3, i + 1] = state[3, i] + dt * state[7, i] + random.gauss(0, Q_t[3, 3])
    state[4, i + 1] = state[4, i] + dt * thrust[i] * np.sin(state[2, i]) * (m_Q + state[8, i] * (np.cos(state[3, i])) ** 2) / (m_Q * (m_Q + state[8, i])) + dt * thrust[i] * np.cos(state[2, i]) * (state[8, i] * np.sin(state[3, i]) * np.cos(state[3, i])) / (m_Q * (m_Q + state[8, i])) + dt * state[8, i] * l * (state[7, i]) ** 2 * np.sin(state[3, i]) / (m_Q + state[8, i]) + random.gauss(0,Q_t[4, 4])
    state[5, i + 1] = state[5, i] + dt * thrust[i] * np.cos(state[2, i]) * (m_Q + state[8, i] * (np.sin(state[3, i])) ** 2) / (m_Q * (m_Q + state[8, i])) + dt * thrust[i] * np.sin(state[2, i]) * (state[8, i] * np.sin(state[3, i]) * np.cos(state[3, i])) / (m_Q * (m_Q + state[8, i])) - dt * state[8, i] * l * (state[7, i]) ** 2 * np.sin(state[3, i]) / (m_Q + state[8, i]) - dt * grav + random.gauss(0, Q_t[5, 5])
    state[6, i + 1] = state[0, i] + dt * tau[i] / I_yy + random.gauss(0, Q_t[6, 6])
    state[7, i + 1] = state[1, i] - dt * thrust[i] * np.sin(state[3, i] - state[2, i]) / (m_Q * l) + random.gauss(0,Q_t[7, 7])
    state[8, i + 1] = state[8, i]

    return state

def Measure(state, y, i):

    y[0, i + 1] = state[0, i + 1] + random.gauss(0, R_t[0, 0])
    y[1, i + 1] = state[1, i + 1] + random.gauss(0, R_t[1, 1])
    y[2, i + 1] = state[2, i + 1] + random.gauss(0, R_t[2, 2])

    return y

def EKF_Predict(mu_t_t, Sigma_t_t, thrust, tau, A_t, i):

    f = np.zeros(n)

    f[0] = mu_t_t[0, i] + dt * mu_t_t[4, i]
    f[1] = mu_t_t[1, i] + dt * mu_t_t[5, i]
    f[2] = mu_t_t[2, i] + dt * mu_t_t[6, i]
    f[3] = mu_t_t[3, i] + dt * mu_t_t[7, i]
    f[4] = mu_t_t[4, i] + dt * thrust[i] * np.sin(mu_t_t[2, i]) * (m_Q + mu_t_t[8, i] * (np.cos(mu_t_t[3, i])) ** 2) / (m_Q * (m_Q + mu_t_t[8, i])) + dt * thrust[i] * np.cos(mu_t_t[2, i]) * (mu_t_t[8, i] * np.sin(mu_t_t[3, i]) * np.cos(mu_t_t[3, i])) / (m_Q * (m_Q + mu_t_t[8, i])) + dt * mu_t_t[8, i] * l * (mu_t_t[7, i]) ** 2 * np.sin(mu_t_t[3, i]) / (m_Q + mu_t_t[8, i])
    f[5] = mu_t_t[5, i] + dt * thrust[i] * np.cos(mu_t_t[2, i]) * (m_Q + mu_t_t[8, i] * (np.sin(mu_t_t[3, i])) ** 2) / (m_Q * (m_Q + mu_t_t[8, i])) + dt * thrust[i] * np.sin(mu_t_t[2, i]) * (mu_t_t[8, i] * np.sin(mu_t_t[3, i]) * np.cos(mu_t_t[3, i])) / (m_Q * (m_Q + mu_t_t[8, i])) - dt * mu_t_t[8, i] * l * (mu_t_t[7, i]) ** 2 * np.sin(mu_t_t[3, i]) / (m_Q + mu_t_t[8, i]) - dt * grav
    f[6] = mu_t_t[0, i] + dt * tau[i] / I_yy
    f[7] = mu_t_t[1, i] - dt * thrust[i] * np.sin(mu_t_t[3, i] - mu_t_t[2, i]) / (m_Q * l)
    f[8] = mu_t_t[8, i]

    mu_t_plus_t = f
    Sigma_t_plus_t = A_t @ Sigma_t_t[:, :, i] @ A_t.T + Q_t

    return mu_t_plus_t, Sigma_t_plus_t

def EKF_Update(mu_t_t, Sigma_t_t, mu_t_plus_t, Sigma_t_plus_t, C_t, y, i):

    g = np.zeros(m)

    g[0] = mu_t_t[0, i]
    g[1] = mu_t_t[1, i]
    g[2] = mu_t_t[2, i]

    mu_t_t[:, i + 1] = mu_t_plus_t + Sigma_t_plus_t @ C_t.T @ np.linalg.inv(C_t @ Sigma_t_plus_t @ C_t.T + R_t) @ (y[:, i + 1] - g)
    Sigma_t_t[:, :, i + 1] = Sigma_t_plus_t - Sigma_t_plus_t @ C_t.T @ np.linalg.inv(C_t @ Sigma_t_plus_t @ C_t.T + R_t) @ C_t @ Sigma_t_plus_t

    return mu_t_t, Sigma_t_t

def Confidence(upper_conf_int, lower_conf_int, mu_t_t, Sigma_t_t, i):

    for j in range(n):
        upper_conf_int[j, i + 1] = mu_t_t[j, i + 1] + 1.96 * np.sqrt(Sigma_t_t[j, j, i + 1])
        lower_conf_int[j, i + 1] = mu_t_t[j, i + 1] - 1.96 * np.sqrt(Sigma_t_t[j, j, i + 1])

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

    A = F.jacobian(X)
    C = G.jacobian(X)

    return A, C


# Simulation loop

[state, thrust, tau, y, mu_t_t, Sigma_t_t, upper_conf_int, lower_conf_int] = Simulation(steps, dt, state_initial, thrust_initial, tau_initial, mu_initial, Sigma_initial)



# Plots

plt.figure(1)
plt.plot(t, state[1], label='True')
plt.plot(t, mu_t_t[1], label='Belief')
plt.fill_between(t, upper_conf_int[1], lower_conf_int[1], color='green', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('Time (sec)')
plt.ylabel('Position in x (m)')
plt.legend()
plt.show()



