import numpy as np
import os
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
PATH_tdi_backtracking_plots = os.path.join(PATH_bethLISA,
                                           "dist/tdi_backtracking_plots/")


INTERFEROMETERS = ["isi", "tmi"]
MOSAS = ["12", "23", "31", "13", "32", "21"]
MOSA_TDI = {
    "12": ("X", "Y"), "23": ("Y", "Z"), "31": ("X", "Z"),
    "13": ("X", "Z"), "32": ("Y", "Z"), "21": ("X", "Y")
}


X_12 = [
    {"coeff": -1, "delay_mosa": []},
    {"coeff": 1, "delay_mosa": ["13", "31"]},
    {"coeff": 1, "delay_mosa": ["12", "21", "13", "31"]},
    {"coeff": -1, "delay_mosa": ["13", "31", "13", "31", "12", "21"]},
]
X_21 = [
    {"coeff": -1, "delay_mosa": ["21"]},
    {"coeff": 1, "delay_mosa": ["21", "13", "31"]},
    {"coeff": 1, "delay_mosa": ["21", "12", "21", "13", "31"]},
    {"coeff": -1, "delay_mosa": ["21", "13", "31", "13", "31", "12", "21"]},
]
X_13 = [
    {"coeff": 1, "delay_mosa": []},
    {"coeff": -1, "delay_mosa": ["12", "21"]},
    {"coeff": -1, "delay_mosa": ["13", "31", "12", "21"]},
    {"coeff": 1, "delay_mosa": ["12", "21", "12", "21", "13", "31"]},
]
X_31 = [
    {"coeff": 1, "delay_mosa": ["31"]},
    {"coeff": -1, "delay_mosa": ["31", "12", "21"]},
    {"coeff": -1, "delay_mosa": ["31", "13", "31", "12", "21"]},
    {"coeff": 1, "delay_mosa": ["31", "12", "21", "12", "21", "13", "31"]},
]
Y_12 = [
    {"coeff": 1, "delay_mosa": ["12"]},
    {"coeff": -1, "delay_mosa": ["12", "23", "32"]},
    {"coeff": -1, "delay_mosa": ["12", "21", "12", "23", "32"]},
    {"coeff": 1, "delay_mosa": ["12", "23", "32", "23", "32", "21", "12"]},
]
Y_21 = [
    {"coeff": 1, "delay_mosa": []},
    {"coeff": -1, "delay_mosa": ["23", "32"]},
    {"coeff": -1, "delay_mosa": ["21", "12", "23", "32"]},
    {"coeff": 1, "delay_mosa": ["23", "32", "23", "32", "21", "12"]},
]
Y_23 = [
    {"coeff": -1, "delay_mosa": []},
    {"coeff": 1, "delay_mosa": ["21", "12"]},
    {"coeff": 1, "delay_mosa": ["23", "32", "21", "12"]},
    {"coeff": -1, "delay_mosa": ["21", "12", "21", "12", "23", "32"]},
]
Y_32 = [
    {"coeff": -1, "delay_mosa": ["32"]},
    {"coeff": 1, "delay_mosa": ["32", "21", "12"]},
    {"coeff": 1, "delay_mosa": ["32", "23", "32", "21", "12"]},
    {"coeff": -1, "delay_mosa": ["32", "21", "12", "21", "12", "23", "32"]},
]
Z_13 = [
    {"coeff": -1, "delay_mosa": ["13"]},
    {"coeff": 1, "delay_mosa": ["13", "32", "23"]},
    {"coeff": 1, "delay_mosa": ["13", "31", "13", "32", "23"]},
    {"coeff": -1, "delay_mosa": ["13", "32", "23", "32", "23", "31", "13"]},
]
Z_31 = [
    {"coeff": -1, "delay_mosa": []},
    {"coeff": 1, "delay_mosa": ["32", "23"]},
    {"coeff": 1, "delay_mosa": ["31", "13", "32", "23"]},
    {"coeff": -1, "delay_mosa": ["32", "23", "32", "23", "31", "13"]},
]
Z_23 = [
    {"coeff": 1, "delay_mosa": ["23"]},
    {"coeff": -1, "delay_mosa": ["23", "31", "13"]},
    {"coeff": -1, "delay_mosa": ["23", "32", "23", "31", "13"]},
    {"coeff": 1, "delay_mosa": ["23", "31", "13", "31", "13", "32", "23"]},
]
Z_32 = [
    {"coeff": 1, "delay_mosa": []},
    {"coeff": -1, "delay_mosa": ["31", "13"]},
    {"coeff": -1, "delay_mosa": ["32", "23", "31", "13"]},
    {"coeff": 1, "delay_mosa": ["31", "13", "31", "13", "32", "23"]},
]


def copy_equation(equation):
    equation_copy = []
    for term in equation:
        term_copy = {}
        term_copy["coeff"] = term["coeff"]
        term_copy["delay_mosa"] = term["delay_mosa"].copy()
        equation_copy.append(term_copy)
    return equation_copy


def apply_coeff(coeff, equation):
    equation_copy = copy_equation(equation)
    for term in equation_copy:
        term["coeff"] *= coeff
    return equation_copy


def apply_delay_mosa(delay_mosa, equation):
    equation_copy = copy_equation(equation)
    for term in equation_copy:
        term["delay_mosa"].insert(0, delay_mosa)
    return equation_copy


def init_gltich_info(glitch_input_txt, glitch_i, t_window):
    glitch_info = np.genfromtxt(PATH_glitch_data + glitch_input_txt)
    t_g = glitch_info[1:, 6][glitch_i]
    t_0 = glitch_info[1:, 5][0]
    dt = glitch_info[1:, 3][0]

    t_range = t_g - t_0 + (-1 * t_window[0], t_window[1])

    return t_range, dt, t_0


def init_simulation_data(simulation_input_h5, interferometer, mosa, t_range, dt, t_0):
    i_range = (int(t_range[0] / dt), int(t_range[1] / dt))
    delay_data = {}

    with h5py.File(PATH_simulation_data + simulation_input_h5, "r") as sim_file:
        sim_data = TimeSeries(
            sim_file[interferometer + "_carrier_fluctuations"][mosa][i_range[0]:i_range[1]],
            delta_t=dt,
            epoch=t_0,
        )
        for mosa in MOSAS:
            delay_data[mosa] = TimeSeries(
                sim_file["mprs"][mosa][i_range[0]:i_range[1]],
                delta_t=dt,
                epoch=t_0,
            )

    return sim_data, delay_data


def init_tdi_data(tdi_input_h5, tdi_channel_A, tdi_channel_B, t_range, dt, t_0):
    i_range = (int(t_range[0] / dt), int(t_range[1] / dt))

    with h5py.File(PATH_tdi_data + tdi_input_h5, "r") as tdi_file:
        tdi_A = TimeSeries(
            tdi_file[tdi_channel_A][i_range[0]:i_range[1]],
            delta_t=dt,
            epoch=t_0,
        )
        tdi_B = TimeSeries(
            tdi_file[tdi_channel_B][i_range[0]:i_range[1]],
            delta_t=dt,
            epoch=t_0,
        )

    return tdi_A, tdi_B


def initialize_data(
    glitch_input_txt, simulation_input_h5, tdi_input_h5, interferometer, mosa,
    tdi_channel, glitch_i, t_window
):
    glitch_info = np.genfromtxt(PATH_glitch_data + glitch_input_txt)
    t_g = glitch_info[1:, 5][glitch_i]
    t_0 = glitch_info[1:, 4][0]
    dt = glitch_info[1:, 2][0]

    t_range = t_g - t_0 + (-1 * t_window[0], t_window[1])
    
    i_range = (int(t_range[0] / dt), int(t_range[1] / dt))

    delay_data = {}

    with h5py.File(PATH_simulation_data + simulation_input_h5, "r") as sim_file:
        sim_data = TimeSeries(
            sim_file[interferometer][mosa][i_range[0]:i_range[1]],
            delta_t=dt,
            epoch=t_0,
        )
        for mosa in MOSAS:
            delay_data[mosa] = TimeSeries(
                sim_file["mprs"][mosa][i_range[0]:i_range[1]],
                delta_t=dt,
                epoch=t_0,
            )

    with h5py.File(PATH_tdi_data + tdi_input_h5, "r") as tdi_file:
        tdi_data = TimeSeries(
            tdi_file[tdi_channel][i_range[0]:i_range[1]],
            delta_t=dt,
            epoch=t_0,
        )

    return sim_data, delay_data, tdi_data, t_g + (-1 * t_window[0], t_window[1])


def data_to_equation(equation, delay_data, t_range):
    # SUM DELAYS
    for term in equation:
        term["delay"] = 0
        for mosa in term["delay_mosa"]:
            term["delay"] += mean(delay_data[mosa][:])
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
    delays.append(t_range[1] - t_range[0])

    return delays, coeff


def piece_by_piece_backtrack(tdi_data, delays, coeff):
    i_delays = [int(delay / tdi_data.delta_t) for delay in delays]
    
    model_vals = [0 for i in range(i_delays[-1])]

    for i_delay in range(0, len(i_delays) - 1):
        for i in range(i_delays[i_delay], i_delays[i_delay + 1]):
            model_vals[i - i_delays[0]] = tdi_data[i]/coeff[0]
            for j in range(1, i_delay + 1):
                model_vals[i - i_delays[0]] -= coeff[j]*model_vals[i-i_delays[j]]/coeff[0]

    model = TimeSeries(
        model_vals,
        delta_t=tdi_data.delta_t,
        epoch=tdi_data.start_time,
    )

    return model


def make_equation(interferometer, mosa, tdi_channel):
    A_ij = globals()[tdi_channel.upper() + "_" + mosa]
    if interferometer.lower() == "isi":
        return A_ij
    else:
        A_ji = globals()[tdi_channel.upper() + "_" + mosa[::-1]]

        return apply_coeff(-1/2, A_ij + apply_delay_mosa(mosa[::-1], A_ji))


def compute_model(
    tdi_data, tdi_channel, interferometer, mosa, delay_data, t_range
):
    equation = make_equation(interferometer, mosa, tdi_channel)

    delays, coeff = data_to_equation(
        equation=equation,
        delay_data=delay_data,
        t_range=t_range,
    )

    model = piece_by_piece_backtrack(
        tdi_data=tdi_data,
        delays=delays,
        coeff=coeff,
    )

    return model


def compute_averaged_model(
    tdi_data_A, tdi_data_B, tdi_channel_A, tdi_channel_B, interferometer, mosa,
    delay_data, t_range
):
    model_A = compute_model(
        tdi_data=tdi_data_A,
        tdi_channel=tdi_channel_A,
        interferometer=interferometer,
        mosa=mosa,
        delay_data=delay_data,
        t_range=t_range,
    )

    model_B = compute_model(
        tdi_data=tdi_data_B,
        tdi_channel=tdi_channel_B,
        interferometer=interferometer,
        mosa=mosa,
        delay_data=delay_data,
        t_range=t_range,
    )

    model = TimeSeries(
        [(model_A[i] + model_B[i]) / 2 for i in range(len(model_A))],
        delta_t=model_A.delta_t,
        epoch=model_A.start_time,
    )

    return model


def compute_overlap(sim_data, model, psd_data_h5):
    psd_fs = load_frequencyseries(PATH_psd_data + psd_data_h5)
    df = psd_fs.delta_f

    sim_data_fs = sim_data.to_frequencyseries(delta_f=df)
    model_fs = model.to_frequencyseries(delta_f=df)

    overlap = matchedfilter.overlap(model_fs, sim_data_fs, psd=psd_fs)

    return round(abs(overlap), 3)


def plot(sim_data, tdi_X, tdi_Y, model, overlap, plot_output):
    figure, axis = plt.subplots(3, 1, figsize=(8, 8), height_ratios=[3, 1, 1])

    axis[0].plot(
        model.sample_times,
        model,
        label="Overlap: " + str(overlap),
    )
    axis[0].plot(
        sim_data.sample_times,
        sim_data,
    )
    axis[0].set_title("Sim and Reconstructions")
    axis[0].legend(loc="upper right")

    axis[1].plot(tdi_X.sample_times, tdi_X)
    axis[1].set_title("TDI X")

    axis[2].plot(tdi_Y.sample_times, tdi_Y)
    axis[2].set_title("TDI Y")

    plt.subplots_adjust(hspace=0.5)

    plt.savefig(PATH_tdi_backtracking_plots + plot_output)


def tdi_backtracking(
    glitch_input_txt, simulation_input_h5, tdi_input_h5,
    plot=False
):
    glitch_info = np.genfromtxt(PATH_glitch_data + glitch_input_txt, dtype=str)
    expected_inj_points = glitch_info[1:, 1]
    num_glitches = len(expected_inj_points)

    predicted_inj_points = []
    
    for glitch_i in range(num_glitches):
        overlaps = {}
        for interferometer in INTERFEROMETERS:
            for mosa in MOSAS:
                t_range, dt, t_0 = init_gltich_info(
                    glitch_input_txt=glitch_input_txt,
                    glitch_i=glitch_i,
                    t_window=(5, 140), # tdi sampling window
                )

                sim_data, delay_data = init_simulation_data(
                    simulation_input_h5=simulation_input_h5,
                    interferometer=interferometer,
                    mosa=mosa,
                    t_range=t_range,
                    dt=dt,
                    t_0=t_0,
                )

                tdi_channel_A = MOSA_TDI[mosa][0]
                tdi_channel_B = MOSA_TDI[mosa][1]

                tdi_data_A, tdi_data_B = init_tdi_data(
                    tdi_input_h5=tdi_input_h5,
                    tdi_channel_A=tdi_channel_A,
                    tdi_channel_B=tdi_channel_B,
                    t_range=t_range,
                    dt=dt,
                    t_0=t_0,
                )

                model = compute_averaged_model(
                    tdi_data_A=tdi_data_A,
                    tdi_data_B=tdi_data_B,
                    tdi_channel_A=tdi_channel_A,
                    tdi_channel_B=tdi_channel_B,
                    interferometer=interferometer,
                    mosa=mosa,
                    delay_data=delay_data,
                    t_range=t_range,
                )

                overlap = compute_overlap(
                    sim_data=sim_data,
                    model=model,
                    psd_data_h5=tdi_channel_A + tdi_channel_B + "_psd.hdf",
                )

                overlaps[interferometer + "_" + mosa] = overlap
                
                if plot:
                    plot(
                        sim_data=sim_data,
                        tdi_X=tdi_data_A,
                        tdi_Y=tdi_data_B,
                        model=model,
                        overlap=overlap,
                        plot_output=interferometer + "backtracking_plot.png",
                    )

        predicted_inj_point = max(overlaps, key=overlaps.get)
        predicted_inj_points.append(predicted_inj_point)

    num_success = 0

    for i in range(len(predicted_inj_points)):
        predicted_inj_point = predicted_inj_points[i]
        expected_inj_point = expected_inj_points[i]

        interferometer = predicted_inj_point[:3]
        mosa = predicted_inj_point[4:]

        if interferometer in expected_inj_point and mosa in expected_inj_point:
            num_success += 1

        print(f"""{i} -- PREDICTED: {predicted_inj_point}
     EXPECTED: {expected_inj_point}""")

    print(f"{num_success}/{num_glitches} inj_points identified successfuly")


if __name__ == "__main__":
    tdi_backtracking(
        glitch_input_txt="default_glitch_output.txt",
        simulation_input_h5="default_simulation_output.h5",
        tdi_input_h5="default_tdi_output.h5",
    )
