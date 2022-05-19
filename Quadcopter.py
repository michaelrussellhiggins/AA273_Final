# Python Packages

import numpy as np
import random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Simulation settings

dt = 0.01   # Simulation time step - sec
t_final = 1   # Simulation run time - sec
steps = round(t_final/dt)
t = np.zeros(steps+1)   # Initializes time vector


# Noise

Q_t = 0.001*np.identity(9)   # Process noise
R_t = 0.001*np.identity(3)   # Measurement noise


# Initialization

# Dynamics

m_Q = 5
m_p = 2
I_yy = 0.01
grav = 9.81
l = 1

m = 3
n = 9

state = np.zeros((n, steps+1))
state[8, 0] = m_p

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


# Control

# The initial control output is currently set to a hovering thrust

thrust = np.zeros(steps+1)   # Thrust vector
tau = np.zeros(steps+1)   # Torque vector

thrust[0] = (m_Q + m_p)*grav   # Initial thrust
tau[0] = 0   # Initial torque

# Measurement

y = np.zeros((3, steps+1))   # Measurement vector

y[:, 0] = np.NaN   # Sets measurements at t=0 to not plot since they do not exist

# EKF Initialization

mu_initial = np.zeros(n)   # Mean of initial guess for state
mu_initial[n-1] = 1;   # Guesses that the quadcopter starts stationary at the origin and has a payload of mass 1 kg
Sigma_initial = 0.1*np.identity(9)   # Covariance of initial guess for state

distribution = multivariate_normal(mu_initial, Sigma_initial)   # Define distribution from which to sample initial
# guess for EKF

mu_t_t = np.zeros((n, steps+1))   # Mean update vector
mu_t_t[:, 0] = distribution.rvs()   # Initial guess for state

Sigma_t_t = np.zeros((n, n, steps+1))   # Prediction update vector
Sigma_t_t[:, :, 0] = Sigma_initial   # Initial guess for covariance

f = np.zeros(n)   # For mean prediction step
g = np.zeros(m)   # For mean update step

A_t = np.zeros((n, n))   # Dynamics Jacobian
C_t = np.zeros((m, n))   # Measurement Jacobian

upper_conf_int = np.zeros((n, steps+1))   # Vector for upper confidence interval
lower_conf_int = np.zeros((n, steps+1))   # Vector for lower confidence interval

for j in range(n):
    upper_conf_int[j, 0] = mu_t_t[j, 0] + 1.96 * np.sqrt(Sigma_t_t[j, j, 0])
    lower_conf_int[j, 0] = mu_t_t[j, 0] - 1.96 * np.sqrt(Sigma_t_t[j, j, 0])

# Simulation loop

for i in range(steps):

    # Moves simulation forward by one time step

    t[i+1] = dt*i

    # Control

    # The control output is currently set to maintain a level hover assuming the quadcopter stays unperturbed

    thrust[i+1] = (m_Q + m_p)*grav
    tau[i+1] = 0

    # Dynamics

    state[0, i + 1] = state[0, i] + dt * state[4, i] + random.gauss(0, Q_t[0, 0])
    state[1, i + 1] = state[1, i] + dt * state[5, i] + random.gauss(0, Q_t[1, 1])
    state[2, i + 1] = state[2, i] + dt * state[6, i] + random.gauss(0, Q_t[2, 2])
    state[3, i + 1] = state[3, i] + dt * state[7, i] + random.gauss(0, Q_t[3, 3])
    state[4, i + 1] = state[4, i] + dt * thrust[i] * np.sin(state[2, i]) * (m_Q + state[8, i] * (np.cos(state[3, i]))**2) / (m_Q * (m_Q + state[8, i])) + dt * thrust[i]*np.cos(state[2, i]) * (state[8, i] * np.sin(state[3, i]) * np.cos(state[3, i])) / (m_Q*(m_Q + state[8, i])) + dt * state[8, i] * l * (state[7, i])**2 * np.sin(state[3, i]) / (m_Q + state[8,i]) + random.gauss(0, Q_t[4, 4])
    state[5, i + 1] = state[5, i] + dt * thrust[i] * np.cos(state[2, i]) * (m_Q + state[8, i] * (np.sin(state[3, i]))**2) / (m_Q * (m_Q + state[8, i])) + dt * thrust[i]*np.sin(state[2, i]) * (state[8, i] * np.sin(state[3, i]) * np.cos(state[3, i])) / (m_Q*(m_Q + state[8, i])) - dt * state[8, i] * l * (state[7, i])**2 * np.sin(state[3, i]) / (m_Q + state[8,i]) - dt*grav + random.gauss(0, Q_t[4, 4])
    state[6, i + 1] = state[0, i] + dt * tau[i] / I_yy + random.gauss(0, Q_t[6, 6])
    state[7, i + 1] = state[1, i] - dt * thrust[i] * np.sin(state[3,i] - state[2,i])/(m_Q * l) + random.gauss(0, Q_t[7, 7])
    state[8, i + 1] = state[8, i]

    # Measurement

    y[0, i + 1] = state[0, i+1] + random.gauss(0, R_t[0, 0])
    y[1, i + 1] = state[1, i+1] + random.gauss(0, R_t[1, 1])
    y[2, i + 1] = state[2, i+1] + random.gauss(0, R_t[2, 2])

    # Jacobians

    A_t[0:9, 0:9] = np.identity(n)
    A_t[0:4, 4:8] = dt*np.identity(4)
    A_t[4, 2] = dt * thrust[i] / (2 * m_Q * (mu_t_t[8, i] + m_Q)) * (mu_t_t[8, i] * np.cos(2 * mu_t_t[3, i] + mu_t_t[2, i]) + (mu_t_t[8, i] + 2 * m_Q) * np.cos(mu_t_t[2, i]))
    A_t[4, 3] = dt * mu_t_t[8, i] / (m_Q * (mu_t_t[8, i] + m_Q)) * (thrust[i] * np.cos(2 * mu_t_t[3, i] + mu_t_t[2, i]) + l * (mu_t_t[7, i])**2 * m_Q * np.cos(mu_t_t[3, i]))
    A_t[4, 7] = 2 * dt * l * mu_t_t[8,i] * mu_t_t[7, i] * np.sin(mu_t_t[3, i]) / (mu_t_t[8, i] + m_Q)
    A_t[4, 8] = dt * np.sin(mu_t_t[3,i]) / ((mu_t_t[8, i] + m_Q)**2) * (thrust[i] * np.cos(mu_t_t[3, i] + mu_t_t[2,i]) + l * (mu_t_t[7, i])**2 * m_Q)
    A_t[5, 2] = dt * thrust[i] / (2 * m_Q * (mu_t_t[8, i] + m_Q)) * (mu_t_t[8, i] * np.sin(2 * mu_t_t[3, i] + mu_t_t[2, i]) - (mu_t_t[8, i] + 2 *m_Q) * np.sin(mu_t_t[2, i]))
    A_t[5, 3] = dt * mu_t_t[8, i] / (m_Q * (mu_t_t[8, i] + m_Q)) * (thrust[i] * np.sin(2*mu_t_t[3, i] + mu_t_t[2, i]) + l * (mu_t_t[7, i])**2 * m_Q * np.sin(mu_t_t[3,i]))
    A_t[5, 7] = -2 * dt * l * mu_t_t[8, i] * mu_t_t[7, i] * np.cos(mu_t_t[3,i]) / (mu_t_t[8,i] + m_Q)
    A_t[5, 8] = -dt * np.cos(mu_t_t[3, i]) / ((mu_t_t[8, i] + m_Q)**2) * (thrust[i]*np.cos(mu_t_t[3, i] + mu_t_t[2, i]) + l * (mu_t_t[7, i])**2 * m_Q)
    A_t[7, 2] = dt * thrust[i] * np.cos(mu_t_t[3, i] - mu_t_t[2, i]) / (m_Q * l)
    A_t[7, 3] = -dt * thrust[i] * np.cos(mu_t_t[3, i] - mu_t_t[2, i]) / (m_Q * l)

    C_t[0:3, 0:3] = np.identity(m)

    f[0] = mu_t_t[0, i] + dt * mu_t_t[4, i]
    f[1] = mu_t_t[1, i] + dt * mu_t_t[5, i]
    f[2] = mu_t_t[2, i] + dt * mu_t_t[6, i]
    f[3] = mu_t_t[3, i] + dt * mu_t_t[7, i]
    f[4] = mu_t_t[4, i] + dt * thrust[i] * np.sin(mu_t_t[2, i]) * (m_Q + mu_t_t[8, i] * (np.cos(mu_t_t[3, i])) ** 2) / (m_Q * (m_Q + mu_t_t[8, i])) + dt * thrust[i] * np.cos(mu_t_t[2, i]) * (mu_t_t[8, i] * np.sin(mu_t_t[3, i]) * np.cos(mu_t_t[3, i])) / (m_Q * (m_Q + mu_t_t[8, i])) + dt * mu_t_t[8, i] * l * (mu_t_t[7, i]) ** 2 * np.sin(mu_t_t[3, i]) / (m_Q + mu_t_t[8, i])
    f[5] = mu_t_t[5, i] + dt * thrust[i] * np.cos(mu_t_t[2, i]) * (m_Q + mu_t_t[8, i] * (np.sin(mu_t_t[3, i])) ** 2) / (m_Q * (m_Q + mu_t_t[8, i])) + dt * thrust[i] * np.sin(mu_t_t[2, i]) * (mu_t_t[8, i] * np.sin(mu_t_t[3, i]) * np.cos(mu_t_t[3, i])) / (m_Q * (m_Q + mu_t_t[8, i])) - dt * mu_t_t[8, i] * l * (mu_t_t[7, i]) ** 2 * np.sin(mu_t_t[3, i]) / (m_Q + mu_t_t[8, i]) - dt * grav
    f[6] = mu_t_t[0, i] + dt * tau[i] / I_yy
    f[7] = mu_t_t[1, i] - dt * thrust[i] * np.sin(mu_t_t[3, i] - mu_t_t[2, i]) / (m_Q * l)
    f[8] = mu_t_t[8, i]

    g[0] = mu_t_t[0, i]
    g[1] = mu_t_t[1, i]
    g[2] = mu_t_t[2, i]

    # EKF Prediction Step

    mu_t_plus_t = f
    Sigma_t_plus_t = A_t @ Sigma_t_t[:, :, i] @ A_t.T + Q_t

    # EKF Update Step

    mu_t_t[:, i + 1] = mu_t_plus_t + Sigma_t_plus_t @ C_t.T @ np.linalg.inv(C_t @ Sigma_t_plus_t @ C_t.T + R_t) @ (y[:, i + 1] - g)
    Sigma_t_t[:, :, i + 1] = Sigma_t_plus_t - Sigma_t_plus_t @ C_t.T @ np.linalg.inv(C_t @ Sigma_t_plus_t @ C_t.T + R_t) @ C_t @ Sigma_t_plus_t

    # Confidence Intervals

    for j in range(n):
        upper_conf_int[j, i + 1] = mu_t_t[j, i + 1] + 1.96 * np.sqrt(Sigma_t_t[j, j, i + 1])
        lower_conf_int[j, i + 1] = mu_t_t[j, i + 1] - 1.96 * np.sqrt(Sigma_t_t[j, j, i + 1])


# Plots

plt.figure(1)
plt.plot(t, state[0], label='True')
plt.plot(t, mu_t_t[0], label='Belief')
plt.fill_between(t, upper_conf_int[0], lower_conf_int[0], color='green', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('Time (sec)')
plt.ylabel('Position in x (m)')
plt.legend()
plt.show()