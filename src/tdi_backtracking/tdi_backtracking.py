import numpy as np
import os
import h5py
import argparse
from statistics import mean
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import FrequencySeries
from scipy.signal import welch
from pycbc.filter import matchedfilter
from performance_analysis import plot_overlap, plot_level_performance

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")
PATH_psd_data = os.path.join(PATH_bethLISA, "dist/psd_data/")
PATH_tdi_backtracking_plots = os.path.join(PATH_bethLISA,
                                           "dist/tdi_backtracking_plots/")
PATH_tdi_backtracking_results = os.path.join(PATH_bethLISA,
                                             "dist/tdi_backtracking_results/")
FREQUENCY_BAND_COEFF = 0.75
INTERFEROMETERS = ["isi", "tmi", "rfi"]
MOSAS = ["12", "23", "31", "13", "32", "21"]
ETA_TDI = {
    "12": ("X", "Y"), "23": ("Y", "Z"), "31": ("X", "Z"),
    "13": ("X", "Z"), "32": ("Y", "Z"), "21": ("X", "Y")
}
RFI_OPPOSITES = {
    "12": "13", "23": "21", "31": "32", "13": "12", "21": "23", "32": "31"
}
TDI_ETA = {
    "X": ["12", "21", "13", "31"], "Y": ["23", "32", "12", "21"],
    "Z": ["31", "13", "23", "32"]
}
RFI_COMBINATION_MID = ["X12", "Y23", "Z31"]
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


def init_cl():
    parser = argparse.ArgumentParser()

    # FILE MANAGEMENT
    parser.add_argument(
        "--glitch_input_txt",
        type=str,
        default="default_glitch_output.txt",
        help="Glitch input txt file name",
    )
    parser.add_argument(
        "--simulation_input_h5",
        type=str,
        default="default_simulation_output.h5",
        help="Simulation input h5 file name",
    )
    parser.add_argument(
        "--tdi_input_h5",
        type=str,
        default="default_tdi_output.h5",
        help="Tdi input h5 file name",
    )

    # Scale
    parser.add_argument(
        "--min_glitch_i",
        type=int,
        default=0,
        help="Start glitch index to run algorithm",
    )
    parser.add_argument(
        "--max_glitch_i",
        type=int,
        help="End glitch index to run algorithm",
    )

    # Process
    parser.add_argument(
        "--process",
        type=str,
        default="0",
        help="Process number",
    )

    return parser.parse_args()


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


def init_tdi_data(tdi_input_h5, tdi_channel, t_range, dt, t_0):
    i_range = (int(t_range[0] / dt), int(t_range[1] / dt))

    with h5py.File(PATH_tdi_data + tdi_input_h5, "r") as tdi_file:
        tdi_data = TimeSeries(
            tdi_file[tdi_channel][i_range[0]:i_range[1]],
            delta_t=dt,
            epoch=t_0,
        )

    return tdi_data


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
    mosa_ij = mosa
    mosa_ji = mosa_ij[::-1]
    mosa_ij_op = RFI_OPPOSITES[mosa_ij]
    mosa_ji_op = mosa_ij_op[::-1]

    A_ij = globals()[tdi_channel.upper() + "_" + mosa_ij]
    A_ji = globals()[tdi_channel.upper() + "_" + mosa_ji]

    if interferometer.lower() == "isi":
        return A_ij
    elif interferometer.lower() == "tmi":
        return apply_coeff(-1/2, A_ij + apply_delay_mosa(mosa_ij, A_ji))
    elif interferometer.lower() == "rfi" and mosa_ij in ["13", "21", "32"]:
        return apply_delay_mosa(mosa_ji, A_ji)
    else:
        if tdi_channel.upper() + mosa in RFI_COMBINATION_MID:
            A_ji_op = globals()[tdi_channel.upper() + "_" + mosa_ji_op]

            ji = apply_delay_mosa(mosa_ji, A_ji)
            ji_op = apply_coeff(-1, apply_delay_mosa(mosa_ji_op, A_ji_op))

            return apply_coeff(1/2, ji + ji_op)
        else:
            return apply_coeff(1/2, A_ij + apply_delay_mosa(mosa_ji, A_ji))


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


def compute_single_model(
    tdi_data, tdi_channel, interferometer, mosa, delay_data, t_range
):
    model = compute_model(
        tdi_data=tdi_data,
        tdi_channel=tdi_channel,
        interferometer=interferometer,
        mosa=mosa,
        delay_data=delay_data,
        t_range=t_range,
    )

    return TimeSeries(
        model,
        delta_t=model.delta_t,
        epoch=model.start_time,
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

    return TimeSeries(
        [(model_A[i] + model_B[i]) / 2 for i in range(len(model_A))],
        delta_t=model_A.delta_t,
        epoch=model_A.start_time,
    )


def compute_averaged_psd(data_segs, num_r, duration, dt):
    seg_len = len(data_segs[0])
    psd = []

    for seg in data_segs:
        seg = seg[:int(duration / dt)]
        seg_f, seg_psd = welch(seg, fs=int(1 / dt), nperseg=int(seg_len))

        if not psd:
            psd = [0 for i in range(len(seg_f))]
        psd = map(lambda x, y: x + y, psd, seg_psd)

    psd = map(lambda x: x / num_r, list(psd))
    delta_f = seg_f[1] - seg_f[0]

    return FrequencySeries(list(psd), delta_f=delta_f)


def seg_data(data, num_r, duration, dt):
    seg_len = int(len(data)/num_r)
    data_segs = []

    for i in range(num_r):
        i_min = i * seg_len
        i_max = i_min + int(duration/dt)

        data_segs.append(data[i_min:i_max])

    return data_segs


def compute_interferometer_psd(interferometer, mosa, num_r, duration, dt, t_0):
    with h5py.File(PATH_simulation_data + "psd_sample_simulation.h5", "r") as sim_file:
        sim_data = TimeSeries(
            sim_file[interferometer + "_carrier_fluctuations"][mosa],
            delta_t=dt,
            epoch=t_0,
        )

    data_segs = seg_data(sim_data, num_r, duration, dt)

    interferometer_psd = compute_averaged_psd(
        data_segs=data_segs,
        num_r=num_r,
        duration=duration,
        dt=dt,
    )

    return interferometer_psd


def compute_model_psd(
    interferometer, mosa, tdi_channel, delay_data, num_r, duration, dt, t_0
):
    with h5py.File(PATH_tdi_data + "psd_sample_tdi.h5", "r") as tdi_file:
        tdi_data = TimeSeries(
            tdi_file[tdi_channel],
            delta_t=dt,
            epoch=t_0,
        )

    data_segs = seg_data(
        data=tdi_data,
        num_r=num_r,
        duration=145, #EDIT
        dt=dt,
    )

    recon_data_segs = []
    for i in range(len(data_segs)):
        recon_data_segs.append(
            compute_single_model(
                tdi_data=data_segs[i],
                tdi_channel=tdi_channel,
                interferometer=interferometer,
                mosa=mosa,
                delay_data=delay_data,
                t_range=(0, 145), #EDIT
            )
        )

    model_psd = compute_averaged_psd(
        data_segs=recon_data_segs,
        num_r=num_r,
        duration=duration,
        dt=dt,
    )

    return model_psd


def compute_outliers(overlaps):
    outliers = {}

    keys = list(overlaps.keys())
    values = list(overlaps.values())

    for i in range(len(keys)):
        omitted = values[i]
        others = values[:i] + values[i+1:]

        average_others = np.average(others)
        std_others = np.std(others)

        outliers[keys[i]] = round(abs((omitted - average_others) / std_others), 3)

    return outliers


def tdi_backtracking(
    glitch_input_txt, simulation_input_h5, tdi_input_h5, min_glitch_i,
    max_glitch_i, output_txt, make_plots=False,
):
    glitch_info_str = np.genfromtxt(PATH_glitch_data + glitch_input_txt, dtype=str)
    glitch_info_int = np.genfromtxt(PATH_glitch_data + glitch_input_txt, dtype=int)
    glitch_info_float = np.genfromtxt(PATH_glitch_data + glitch_input_txt, dtype=float)
    expected_inj_points = glitch_info_str[1:, 1]
    levels = glitch_info_float[1:, 7]
    t_rises = glitch_info_int[1:, 8]
    t_falls = glitch_info_int[1:, 9]
    num_glitches = len(expected_inj_points)

    if not max_glitch_i:
        max_glitch_i = num_glitches

    if os.path.exists(PATH_tdi_backtracking_results + output_txt):
        os.remove(PATH_tdi_backtracking_results + output_txt)

    num_success = 0

    for glitch_i in range(min_glitch_i, max_glitch_i):
        overlaps = {}
        for interferometer in INTERFEROMETERS:
            for mosa in MOSAS:
                tdi_overlaps = []
                for tdi_channel in ETA_TDI[mosa]:
                    t_range, dt, t_0 = init_gltich_info(
                        glitch_input_txt=glitch_input_txt,
                        glitch_i=glitch_i,
                        t_window=(5, 140), # tdi sampling window in s
                    )

                    sim_data, delay_data = init_simulation_data(
                        simulation_input_h5=simulation_input_h5,
                        interferometer=interferometer,
                        mosa=mosa,
                        t_range=t_range,
                        dt=dt,
                        t_0=t_0,
                    )

                    tdi_data = init_tdi_data(
                        tdi_input_h5=tdi_input_h5,
                        tdi_channel=tdi_channel,
                        t_range=t_range,
                        dt=dt,
                        t_0=t_0,
                    )

                    model = compute_single_model(
                        tdi_data=tdi_data,
                        tdi_channel=tdi_channel,
                        interferometer=interferometer,
                        mosa=mosa,
                        delay_data=delay_data,
                        t_range=t_range,
                    )

                    duration = (t_rises[glitch_i] + t_falls[glitch_i]) * 3
                    sim_data = sim_data[:int(duration/dt)]
                    model = model[:int(duration/dt)]

                    num_r = 25 # number of noise realizations for psd estimations

                    interferometer_noise_psd = compute_interferometer_psd(
                        interferometer=interferometer,
                        mosa=mosa,
                        num_r=num_r,
                        duration=duration,
                        dt=dt,
                        t_0=t_0,
                    )

                    model_noise_psd = compute_model_psd(
                        interferometer=interferometer,
                        mosa=mosa,
                        tdi_channel=tdi_channel,
                        delay_data=delay_data,
                        num_r=num_r,
                        duration=duration,
                        dt=dt,
                        t_0=t_0,
                    )

                    idx_cutoff = int(len(model_noise_psd.get_sample_frequencies()) * FREQUENCY_BAND_COEFF)

                    sim_data_fs = sim_data.to_frequencyseries(delta_f=model_noise_psd.delta_f)[:idx_cutoff]
                    model_fs = model.to_frequencyseries(delta_f=model_noise_psd.delta_f)[:idx_cutoff]
                    psd = (interferometer_noise_psd + model_noise_psd)[:idx_cutoff]

                    overlap = round(abs(matchedfilter.overlap(model_fs, sim_data_fs, psd=psd)), 3)

                    tdi_overlaps.append(overlap)

                    if make_plots:
                        plot_overlap(
                            sim_data=sim_data,
                            sim_data_fs=sim_data_fs,
                            tdi_data=tdi_data,
                            tdi_channel=tdi_channel,
                            model=model,
                            model_fs=model_fs,
                            psd=psd,
                            overlap=overlap,
                            plot_output=f"{glitch_i}_{interferometer}_{mosa}_{tdi_channel}.png",
                        )

                overlaps[interferometer + "_" + mosa] = np.average(tdi_overlaps)

        outliers = compute_outliers(overlaps)

        predicted_inj_point = max(outliers, key=outliers.get)
        expected_inj_point = expected_inj_points[glitch_i]
        expected_inj_point = f"{expected_inj_point[8:11]}_{expected_inj_point[-2:]}"

        if predicted_inj_point == expected_inj_point:
            identified = True
            num_success += 1
        else:
            identified = False
            print(f"{glitch_i} -- PREDICTED: {predicted_inj_point}, EXPECTED: {expected_inj_point}, LEVEL: {levels[glitch_i]}")

        with open(PATH_tdi_backtracking_results + output_txt, "a") as f:
            output = f"{glitch_i} {levels[glitch_i]} {identified} {predicted_inj_point} {expected_inj_point} "
            for key, value in outliers.items():
                output += f"{value} "
            f.write(output[:-1] + "\n")

        print(f"{num_success}/{(max_glitch_i - min_glitch_i)} inj_points identified successfuly so far from set of {num_glitches} points")


if __name__ == "__main__":
    cl_args = init_cl()

    process = cl_args.process

    tdi_backtracking(
        glitch_input_txt=cl_args.glitch_input_txt,
        simulation_input_h5=cl_args.simulation_input_h5,
        tdi_input_h5=cl_args.tdi_input_h5,
        make_plots=True,
        min_glitch_i=cl_args.min_glitch_i,
        max_glitch_i=cl_args.max_glitch_i,
        output_txt=process + "results.txt",
    )

    # name = "gw_test"
    # tdi_backtracking(
    #     glitch_input_txt=name + ".txt",
    #     simulation_input_h5=name + ".h5",
    #     tdi_input_h5=name + ".h5",
    #     make_plots=True,
    #     min_glitch_i=cl_args.min_glitch_i,
    #     max_glitch_i=cl_args.max_glitch_i,
    #     output_txt=process + "results.txt",
    # )
