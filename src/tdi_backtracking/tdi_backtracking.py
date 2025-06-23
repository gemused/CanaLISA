import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from statistics import mean
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.types.timeseries import TimeSeries
from pycbc.filter import matchedfilter

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")
PATH_psd_data = os.path.join(PATH_bethLISA, "dist/psd_data/")

MOSAS = ["12", "23", "31", "13", "32", "21"]
TMI_12_X = [
    {"coeff": 1/2, "delay_mosa": [], "delay": 0},
    {"coeff": 1/2, "delay_mosa": ["21", "21"], "delay": 0},
    {"coeff": -1/2, "delay_mosa": ["13", "31"], "delay": 0},
    {"coeff": -1/2, "delay_mosa": ["21", "21", "13", "31"], "delay": 0},
    {"coeff": -1/2, "delay_mosa": ["12", "21", "13", "31"], "delay": 0},
    {"coeff": -1/2, "delay_mosa": ["21", "21", "12", "21", "13", "31"],
        "delay": 0},
    {"coeff": 1/2, "delay_mosa": ["13", "31", "13", "31", "12", "21"],
        "delay": 0},
    {"coeff": 1/2, "delay_mosa": ["21", "21", "13", "31", "13", "31", "12", "21"],
        "delay": 0},
]


def initialize_data(
    glitch_input_txt, simulation_input_h5, tdi_input_h5, glitch_i,
    t_window_left, t_window_right
):
    glitch_info = np.genfromtxt(PATH_glitch_data + glitch_input_txt)
    t_g = glitch_info[1:, 5][glitch_i]
    t_0 = glitch_info[1:, 4][0]
    dt = glitch_info[1:, 2][0]

    i_g = int((t_g - t_0) / dt)
    i_window_left = int(t_window_left / dt)
    i_window_right = int(t_window_right / dt)
    i_min = i_g - i_window_left
    i_max = i_g + i_window_right

    delay_data = {}
    with h5py.File(PATH_simulation_data + simulation_input_h5, "r") as sim_file:
        sim_data = sim_file["tmi_carrier_fluctuations"]["12"][i_min:i_max]
        for mosa in MOSAS:
            delay_data[mosa] = sim_file["mprs"][mosa][i_min:i_max] / dt

    with h5py.File(PATH_tdi_data + tdi_input_h5, "r") as tdi_file:
        tdi_data = tdi_file["X"][i_min:i_max]

    return sim_data, delay_data, tdi_data, i_g, i_min, i_max, dt


def data_to_equation(equation, delay_data, i_size):
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

    return delays, coeff


def piece_by_piece_backtracking(tdi_data, delays, coeff):
    model = [0 for i in range(0, delays[-1])]

    for i in range(0, len(delays) - 1):
        for t in range(delays[i], delays[i + 1]):
            model[t] = tdi_data[t]/coeff[0]
            for j in range(1, i + 1):
                model[t] = model[t] - coeff[j]*model[t-delays[j]]/coeff[0]

    return model


def plot_data(
    plot_output, sim_data, model, tdi_data, plt_window_left, plt_window_right,
    i_g, i_min, i_max, dt
):
    i_plt_window_left = int(plt_window_left / dt)
    i_plt_window_right = int(plt_window_right / dt)

    i_plt_min = i_g - i_plt_window_left
    i_plt_max = i_g + i_plt_window_right

    i_data_min = i_g - i_min - i_plt_window_left
    i_data_max = i_g - i_min + i_plt_window_right

    figure, axis = plt.subplots(2, 1, figsize=(8, 6))

    axis[0].plot(range(i_plt_min, i_plt_max), model[i_data_min:i_data_max])
    axis[0].plot(range(i_plt_min, i_plt_max), sim_data[i_data_min:i_data_max])
    axis[0].set_title("Sim (Orange) and Reconstruction (Blue)")

    axis[1].plot(range(i_min, i_max), tdi_data)
    axis[1].set_title("TDI")

    plt.subplots_adjust(hspace=0.3)

    plt.savefig(plot_output)

    return i_data_min, i_data_max


def compute_overlap(psd_data_h5, sim_data, model, i_data_min, i_data_max, dt):
    fs_psd = load_frequencyseries(PATH_psd_data + psd_data_h5)
    df = fs_psd.delta_f

    ts_pre = TimeSeries(model[i_data_min:i_data_max], delta_t=dt)
    fs_pre = ts_pre.to_frequencyseries(delta_f=df)

    ts_sim = TimeSeries(sim_data[i_data_min:i_data_max], delta_t=dt)
    fs_sim = ts_sim.to_frequencyseries(delta_f=df)

    overlap = matchedfilter.overlap(fs_sim, fs_pre, psd=fs_psd)

    return overlap


def tdi_backtracking(
    glitch_input_txt, simulation_input_h5, tdi_input_h5, equation, glitch_i,
    plot_output
):
    sim_data, delay_data, tdi_data, i_g, i_min, i_max, dt = initialize_data(
        glitch_input_txt="default_glitch_output.txt",
        simulation_input_h5="default_simulation_output.h5",
        tdi_input_h5="default_tdi_output.h5",
        glitch_i=glitch_i,
        t_window_left=5,
        t_window_right=140,
    )

    delays, coeff = data_to_equation(
        equation=equation,
        delay_data=delay_data,
        i_size=i_max-i_min,
    )

    model = piece_by_piece_backtracking(
        tdi_data=tdi_data,
        delays=delays,
        coeff=coeff,
    )

    i_data_min, i_data_max = plot_data(
        plot_output=plot_output,
        sim_data=sim_data,
        model=model,
        tdi_data=tdi_data,
        plt_window_left=5,
        plt_window_right=30,
        i_g=i_g,
        i_min=i_min,
        i_max=i_max,
        dt=dt,
    )

    overlap = compute_overlap(
        psd_data_h5="tmi_psd.hdf",
        sim_data=sim_data,
        model=model,
        i_data_min=i_data_min,
        i_data_max=i_data_max,
        dt=dt,
    )

    print(f"Overlap = {overlap}")


if __name__ == "__main__":
    tdi_backtracking(
        glitch_input_txt="default_glitch_output.txt",
        simulation_input_h5="default_simulation_output.h5",
        tdi_input_h5="default_tdi_output.h5",
        plot_output="backtracking_output.png",
        equation=TMI_12_X,
        glitch_i=0
    )
