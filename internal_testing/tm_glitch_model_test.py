import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from scipy.integrate import quad

# Weiner Process Stuff
t_max = 2000 # duration of process
t_min = 0
dt = 1
n = t_max/dt # Number of discrete timesteps

t1 = np.linspace(t_min, t_max, int(n))
# Glitch Stuff
t_inj = 100
t_fall = 200
t_rise = 10
A = 1
wavelength = 1024*10**(-9)
L = 8.33

index = -1
dW = np.sqrt(dt) * np.random.normal(size=int(n))
W = np.cumsum(dW)
window = 5000

def g_FRED(t):
    value = A*(t-t_inj)/(t_fall**2)*2.71828**(-(t-t_inj)/t_fall)
    # value = A*(2.71828**(-(t-t_inj)/(t_rise))-2.71828**(-(t-t_inj)/(t_fall)))/(t_rise-t_fall)
    if value < 0:
        return 0
    else:
        return value

def wiener(t):
    global index
    return W[index]

def n_filtered(t):
    return quad(wiener, t-window, t)[0]/window

def g_FREDS(t):
    # global index
    # index += 1
    return g_FRED(t)#*n_filtered(t)

def v_FREDS(t):
    return quad(g_FREDS, 0, t)[0]

def tmi(t):
    # global index
    # index += 1
    return 2 * v_FREDS(t) / wavelength

def X(t):
    # global index
    # index += 1
    return (v_FREDS(t - 8*L)-2*v_FREDS(t - 4*L)+v_FREDS(t)) / wavelength

Xvals = list(map(X, t1))
TMIvals = list(map(tmi, t1))

figure, axis = plt.subplots(2, 1)

axis[0].plot(t1, Xvals)
axis[1].plot(t1, TMIvals)

plt.savefig("tm_glitch_model_test_output.png")
