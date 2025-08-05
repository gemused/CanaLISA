import numpy as np
import os
import h5py
import argparse
from statistics import mean
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import FrequencySeries
from scipy.signal import welch
from pycbc.filter import matchedfilter
from performance_analysis import plot

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/lisa_data/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/lisa_data/tdi_data/")
PATH_tdi_backtracking_plots = os.path.join(PATH_bethLISA,
                                           "dist/tdi_backtracking/plots/")
PATH_tdi_backtracking_results = os.path.join(PATH_bethLISA,
                                             "dist/tdi_backtracking/results/")
PATH_anomaly_data = os.path.join(PATH_bethLISA, "dist/tdi_backtracking/anomaly_data/")
PATH_pipe_data = os.path.join(PATH_bethLISA, "dist/pipe/pipe_data/")
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
        "--anomaly_input_fn",
        type=str,
        default="default_anomaly_output",
        help="Anomoly input file name (excluding file extension)",
    )
    parser.add_argument(
        "--simulation_input_fn",
        type=str,
        default="default_simulation_output",
        help="Simulation input file name (excluding file extension)",
    )
    parser.add_argument(
        "--tdi_input_fn",
        type=str,
        default="default_tdi_output",
        help="Tdi input file name (excluding file extension)",
    )
    parser.add_argument(
        "--pipe_input_fn",
        type=str,
        default="default_pipe_output",
        help="Pipeline input file name (excluding file extension)",
    )

    # Scale
    parser.add_argument(
        "--min_anomaly_i",
        type=int,
        default=0,
        help="Start anomaly index to run algorithm",
    )
    parser.add_argument(
        "--max_anomaly_i",
        type=int,
        help="End anomaly index to run algorithm",
    )

    # Process
    parser.add_argument(
        "--process",
        type=str,
        default="0",
        help="Process number",
    )

    return parser.parse_args()


def init_pipe_info(pipe_input_fn, anomaly_input_fn):
    anomaly_info = np.genfromtxt(PATH_anomaly_data + anomaly_input_fn + ".txt")
    pipe_info = np.genfromtxt(PATH_pipe_data + pipe_input_fn + ".txt")

    t0 = anomaly_info[1:, 5][0]
    dt = anomaly_info[1:, 3][0]

    return dt, t0


def init_simulation_data(simulation_input_fn, interferometer, mosa, t_range, dt, t0):
    i_range = (int(t_range[0] / dt), int(t_range[1] / dt))
    delay_data = {}

    with h5py.File(PATH_simulation_data + simulation_input_fn + ".h5", "r") as sim_file:
        sim_data = TimeSeries(
            sim_file[interferometer + "_carrier_fluctuations"][mosa][i_range[0]:i_range[1]],
            delta_t=dt,
            epoch=t0,
        )
        for mosa in MOSAS:
            delay_data[mosa] = TimeSeries(
                sim_file["mprs"][mosa][i_range[0]:i_range[1]],
                delta_t=dt,
                epoch=t0,
            )

    return sim_data, delay_data


def init_tdi_data(tdi_input_fn, tdi_channel, t_range, dt, t0):
    i_range = (int(t_range[0] / dt), int(t_range[1] / dt))

    with h5py.File(PATH_tdi_data + tdi_input_fn + ".h5", "r") as tdi_file:
        tdi_data = TimeSeries(
            tdi_file[tdi_channel][i_range[0]:i_range[1]],
            delta_t=dt,
            epoch=t0,
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
    psd = []

    max_delta_f = 0.00125
    nperseg = int(1/(dt * max_delta_f))

    for seg in data_segs:
        # seg = seg[:int(duration / dt)]
        seg_f, seg_psd = welch(seg, fs=int(1 / dt), nperseg=nperseg)

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


def compute_interferometer_psd(interferometer, mosa, num_r, duration, dt, t0):
    with h5py.File(PATH_simulation_data + "psd_sample_simulation.h5", "r") as sim_file:
        sim_data = TimeSeries(
            sim_file[interferometer + "_carrier_fluctuations"][mosa],
            delta_t=dt,
            epoch=t0,
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
    interferometer, mosa, tdi_channel, delay_data, num_r, duration, dt, t0
):
    with h5py.File(PATH_tdi_data + "psd_sample_tdi.h5", "r") as tdi_file:
        tdi_data = TimeSeries(
            tdi_file[tdi_channel],
            delta_t=dt,
            epoch=t0,
        )

    data_segs = seg_data(
        data=tdi_data,
        num_r=num_r,
        duration=duration, #EDIT
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
                t_range=(0, duration), #EDIT
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
    anomaly_input_fn, simulation_input_fn, tdi_input_fn, pipe_input_fn,
    min_anomaly_i, max_anomaly_i, results_output_fn, make_plots=False,
):
    anomaly_data_path = PATH_anomaly_data + anomaly_input_fn + ".txt"
    pipe_data_path = PATH_pipe_data + pipe_input_fn + ".txt"
    results_output_path = PATH_tdi_backtracking_results + results_output_fn + ".txt"

    anomaly_data_str = np.genfromtxt(anomaly_data_path, dtype=str)
    anomaly_data_float = np.genfromtxt(anomaly_data_path, dtype=float)
    pipe_data_float = np.genfromtxt(pipe_data_path, dtype=float)

    t0 = pipe_data_float[0]
    dt = pipe_data_float[1]

    expected_inj_points = anomaly_data_str[0:, 1]
    t_injs = anomaly_data_float[0:, 2]
    levels = anomaly_data_float[0:, 3]

    num_anomalies = len(expected_inj_points)

    if not max_anomaly_i:
        max_anomaly_i = num_anomalies

    if os.path.exists(results_output_path):
        os.remove(results_output_path)

    for anomaly_i in range(min_anomaly_i, max_anomaly_i):
        overlaps = {}
        t_inj = t_injs[anomaly_i]
        for interferometer in INTERFEROMETERS:
            for mosa in MOSAS:
                tdi_overlaps = []
                for tdi_channel in ETA_TDI[mosa]:
                    sample_duration = 400
                    t_range = t_inj - t0 + (0, sample_duration)

                    sim_data, delay_data = init_simulation_data(
                        simulation_input_fn=simulation_input_fn,
                        interferometer=interferometer,
                        mosa=mosa,
                        t_range=t_range,
                        dt=dt,
                        t0=t0,
                    )

                    tdi_data = init_tdi_data(
                        tdi_input_fn=tdi_input_fn,
                        tdi_channel=tdi_channel,
                        t_range=t_range,
                        dt=dt,
                        t0=t0,
                    )

                    model = compute_single_model(
                        tdi_data=tdi_data,
                        tdi_channel=tdi_channel,
                        interferometer=interferometer,
                        mosa=mosa,
                        delay_data=delay_data,
                        t_range=t_range,
                    )

                    duration = sample_duration
                    # sim_data = sim_data[:int(duration/dt)]
                    # model = model[:int(duration/dt)]

                    num_r = 10 # number of noise realizations for psd estimations

                    interferometer_noise_psd = compute_interferometer_psd(
                        interferometer=interferometer,
                        mosa=mosa,
                        num_r=num_r,
                        duration=duration,
                        dt=dt,
                        t0=t0,
                    )

                    model_noise_psd = compute_model_psd(
                        interferometer=interferometer,
                        mosa=mosa,
                        tdi_channel=tdi_channel,
                        delay_data=delay_data,
                        num_r=num_r,
                        duration=duration,
                        dt=dt,
                        t0=t0,
                    )

                    idx_cutoff = int(len(model_noise_psd.get_sample_frequencies()) * FREQUENCY_BAND_COEFF)

                    sim_data_fs = sim_data.to_frequencyseries(delta_f=model_noise_psd.delta_f)[:idx_cutoff]
                    model_fs = model.to_frequencyseries(delta_f=model_noise_psd.delta_f)[:idx_cutoff]
                    psd = (interferometer_noise_psd + model_noise_psd)[:idx_cutoff]
                    # psd = model_noise_psd[:idx_cutoff]

                    overlap = round(abs(matchedfilter.overlap(model_fs, sim_data_fs, psd=psd)), 3)

                    tdi_overlaps.append(overlap)

                    if make_plots:
                        plot(
                            sim_data=sim_data,
                            sim_data_fs=sim_data_fs,
                            tdi_data=tdi_data,
                            tdi_channel=tdi_channel,
                            model=model,
                            model_fs=model_fs,
                            psd=psd,
                            overlap=overlap,
                            plot_output=f"{anomaly_i}_{interferometer}_{mosa}_{tdi_channel}.png",
                        )

                overlaps[interferometer + "_" + mosa] = np.average(tdi_overlaps)

        outliers = compute_outliers(overlaps)

        predicted_inj_point = max(outliers, key=outliers.get)
        expected_inj_point = expected_inj_points[anomaly_i]
        if expected_inj_point != "gw":
            expected_inj_point = f"{expected_inj_point[8:11]}_{expected_inj_point[-2:]}"

        with open(results_output_path, "a") as f:
            output = f"{anomaly_i} {levels[anomaly_i]} {predicted_inj_point} {expected_inj_point} "
            for key, value in outliers.items():
                output += f"{value} "
            f.write(output[:-1] + "\n")

        print(f"Anomaly {anomaly_i - min_anomaly_i} processed from subset [{min_anomaly_i},{max_anomaly_i - 1}] from parent set of {num_anomalies} points")


if __name__ == "__main__":
    cl_args = init_cl()

    tdi_backtracking(
        anomaly_input_fn=cl_args.anomaly_input_fn,
        simulation_input_fn=cl_args.simulation_input_fn,
        tdi_input_fn=cl_args.tdi_input_fn,
        pipe_input_fn=cl_args.pipe_input_fn,
        make_plots=True,
        min_anomaly_i=cl_args.min_anomaly_i,
        max_anomaly_i=cl_args.max_anomaly_i,
        results_output_fn=cl_args.process + "results",
    )

    # name = "gw_test"
    # tdi_backtracking(
    #     anomaly_input_txt=name + ".txt",
    #     simulation_input_h5=name + ".h5",
    #     tdi_input_h5=name + ".h5",
    #     make_plots=True,
    #     min_anomaly_i=cl_args.min_anomaly_i,
    #     max_anomaly_i=cl_args.max_anomaly_i,
    #     output_txt=process + "results.txt",
    # )
