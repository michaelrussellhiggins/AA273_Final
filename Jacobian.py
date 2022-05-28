import sympy as sp
import numpy as np

x = sp.symbols('x')
z = sp.symbols('z')
theta = sp.symbols('theta')
phi = sp.symbols('phi')
x_dot = sp.symbols('x_dot')
z_dot = sp.symbols('z_dot')
theta_dot = sp.symbols('theta_dot')
phi_dot = sp.symbols('phi_dot')
m_p = sp.symbols('m_p')

f = sp.symbols('f')
tau = sp.symbols('tau')
m_Q = 5
l = 1
g = 9.81
I_yy = 0.01
dt = 0.01

eqn1 = x + dt*x_dot
eqn2 = z + dt*z_dot
eqn3 = theta + dt*theta_dot
eqn4 = phi + dt*phi_dot
eqn5 = x_dot + dt*( (m_Q + m_p*(sp.cos(phi))**2)/(m_Q*(m_Q + m_p))*f*sp.sin(theta) + (m_p*sp.sin(phi)*sp.cos(phi))/(m_Q*(m_Q + m_p))*f*sp.cos(theta) + (m_p*l*phi_dot**2*sp.sin(phi))/(m_Q + m_p) )
eqn6 = z_dot + dt*( (m_Q + m_p*(sp.sin(phi))**2)/(m_Q*(m_Q + m_p))*f*sp.cos(theta) + (m_p*sp.sin(phi)*sp.cos(phi))/(m_Q*(m_Q + m_p))*f*sp.sin(theta) - (m_p*l*phi_dot**2*sp.cos(phi))/(m_Q + m_p) - g)
eqn7 = theta_dot + dt*tau/I_yy
eqn8 = phi_dot - dt*f*sp.sin(phi - theta)/(m_Q*l)
eqn9 = m_p

F = sp.Matrix([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9])
G = sp.Matrix([x, z, theta])
X = sp.Matrix([x, z, theta, phi, x_dot, z_dot, theta_dot, phi_dot, m_p])

A_t = F.jacobian(X)
C_t = G.jacobian(X)