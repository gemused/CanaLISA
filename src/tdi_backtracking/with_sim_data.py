import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from statistics import mean

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")

MOSAS = ["12", "23", "31", "13", "32", "21"]

# ~~~~~~~~~~ INITIALIZE DATA ~~~~~~~~~~
glitch_info = np.genfromtxt(PATH_glitch_data + "default_glitch_output.txt")
glitch_i = 0
t_g = glitch_info[1:,5][glitch_i]
t_0 = glitch_info[1:,4][0]
dt = glitch_info[1:,2][0]
t_window_left = 5
t_window_right = 70

i_g = int((t_g - t_0) / dt)
i_window_left = int(t_window_left / dt)
i_window_right = int(t_window_right / dt)
i_min = i_g - i_window_left
i_max = i_g + i_window_right
i_size = i_max - i_min

delay_data = {}
with h5py.File(PATH_simulation_data + "default_simulation_output.h5", "r") as sim_file:
    sim_data = sim_file["tmi_carrier_fluctuations"]["12"][i_min:i_max]
    for mosa in MOSAS:
        delay_data[mosa] = sim_file["mprs"][mosa][i_min:i_max] / dt

with h5py.File(PATH_tdi_data + "default_tdi_output.h5", "r") as tdi_file:
    tdi_data = tdi_file["X"][i_min:i_max]

# ~~~~~~~~~~ INITIALIZE EQUATION DATA ~~~~~~~~~~
equation = [
    {"coeff": 1/2, "delay_mosa": [], "delay": 0},
    {"coeff": 1/2, "delay_mosa": ["21", "21"], "delay": 0},
    {"coeff": -1/2, "delay_mosa": ["13", "31"], "delay": 0},
    {"coeff": -1/2, "delay_mosa": ["21", "21", "13", "31"], "delay": 0},
    {"coeff": -1/2, "delay_mosa": ["12", "21", "13", "31"], "delay": 0},
    {"coeff": -1/2, "delay_mosa": ["21", "21", "12", "21", "13", "31"], "delay": 0},
    {"coeff": 1/2, "delay_mosa": ["13", "31", "13", "31", "12", "21"], "delay": 0},
    {"coeff": 1/2, "delay_mosa": ["21", "21", "13", "31", "13", "31", "12", "21"], "delay": 0},
]

# SUM DELAYS
for term in equation:
    for mosa in term["delay_mosa"]:
        term["delay"] += mean(delay_data[mosa])
    term["delay"] = int(term["delay"])

equation.sort(key=lambda term: term["delay"])

# SUM LIKE-TERMS
i = 0
while i < len(equation) - 1:
    if equation[i]["delay"] == equation[i + 1]["delay"]:
        equation[i]["coeff"] += equation[i + 1]["coeff"]
        equation.remove(equation[i + 1])
    i += 1

coeff = [term["coeff"] for term in equation]
delays = [term["delay"] for term in equation]
delays.append(i_size)

# ~~~~~~~~~~ PIECE-BY-PIECE BACKTRACKING ~~~~~~~~~~

pre = [0 for i in range(0, i_size)]
for i in range(0, len(delays) - 1):
    for t in range(delays[i], delays[i+1]):
        pre[t] = tdi_data[t]/coeff[0]
        for j in range(1, i+1):
            pre[t] = pre[t] - coeff[j]*pre[t-delays[j]]/coeff[0]

# ~~~~~~~~~~ PLOT DATA ~~~~~~~~~~
plt_window_left = 5
plt_window_right = 5
i_plt_window_left = int(plt_window_left / dt)
i_plt_window_right = int(plt_window_right / dt)

i_plt_min = i_g - i_plt_window_left
i_plt_max = i_g + i_plt_window_right

i_plt_size = i_plt_window_left + i_plt_window_right
i_data_min = i_g - i_min - i_plt_window_left
i_data_max = i_g - i_min + i_plt_window_right

figure, axis = plt.subplots(2, 1, figsize=(8,6))

axis[0].plot(range(i_plt_min, i_plt_max), pre[i_data_min:i_data_max])
axis[0].plot(range(i_plt_min, i_plt_max), sim_data[i_data_min:i_data_max])
axis[0].set_title("Sim (Orange) and Reconstruction (Blue)")

axis[1].plot(range(i_min, i_max), tdi_data)
axis[1].set_title("TDI")

plt.subplots_adjust(hspace=0.3)

plt.savefig("with_sim_data_output.png")
