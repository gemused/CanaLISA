# import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

t_min = 0
t_max = 2000
t_vals = [i for i in range(t_min, t_max)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
t_inj = 100
t_fall = 200
t_rise = 10
A = 1

wavelength = 1024*10**(-9)
L = 8

index = -1
dW = np.random.normal(size=int(t_max))
W = np.cumsum(dW)


def g_FRED(t):
    value = A*(t-t_inj)/(t_fall**2)*2.71828**(-(t-t_inj)/t_fall)
    if value < 0:
        return 0
    else:
        return value


def g_FREDS(t):
    return g_FRED(t)


def v_FREDS(t):
    return quad(g_FREDS, 0, t)[0]


def tmi(t):
    return 2 * v_FREDS(t) / wavelength


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


delays = [t_min, 4*L, 8*L, t_max]
coeff = [1, -2, 1]

tmi_vals = [tmi(i) for i in t_vals]

post = [0] * len(t_vals)
for t in t_vals:
    for i in range(0, len(delays) - 1):
        post[t] += coeff[i]*tmi(t-delays[i])

pre = [0 for i in range(t_min, t_max)]
for i in range(0, len(delays) - 1):
    for t in range(delays[i], delays[i+1]):
        pre[t] = post[t]/coeff[0]
        for j in range(1, i+1):
            pre[t] = pre[t] - coeff[j]*pre[t-delays[j]]/coeff[0]

figure, axis = plt.subplots(2, 1, figsize=(8,6))

axis[0].plot(t_vals, post)
axis[0].set_title("X")

axis[1].plot(t_vals, tmi_vals)
axis[1].plot(t_vals, pre)
axis[1].set_title("TMI and Reconstruction")

plt.subplots_adjust(hspace=0.6)

plt.savefig("output.png")
