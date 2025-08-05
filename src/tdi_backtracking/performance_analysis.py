"""
Filename: performance_analysis.py
Author: William Mostrenko
Created: 2025-07-16
Description: Analysis tools for tdi-backtracking performance.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_tdi_backtracking_plots = os.path.join(PATH_bethLISA,
                                           "dist/tdi_backtracking/plots/")
PATH_tdi_backtracking_results = os.path.join(PATH_bethLISA,
                                             "dist/tdi_backtracking/results/")
PATH_tdi_backtracking_performance_plots = os.path.join(PATH_bethLISA,
                                                       "dist/tdi_backtracking/performance_plots/")
INTERFEROMETERS = ["isi", "tmi", "rfi"]
MOSAS = ["12", "23", "31", "13", "32", "21"]
GLITCH_CANDIDATE_THRESHOLD = 8


def plot(
    sim_data, sim_data_fs, tdi_data, tdi_channel, model, model_fs, psd,
    overlap, plot_output
):
    figure, axis = plt.subplots(3, 1, figsize=(8, 8), height_ratios=[3, 3, 1])

    axis[0].plot(
        model.sample_times,
        model,
        label="Model TS",
    )
    axis[0].plot(
        sim_data.sample_times,
        sim_data,
        label="Interferometer TS",
    )
    axis[0].set_title("Sim and Reconstructions TS")
    axis[0].legend(loc="upper right")

    axis[1].plot(
        model_fs.get_sample_frequencies(),
        model_fs,
        label="Model FS Overlap: " + str(overlap),
    )
    axis[1].plot(
        sim_data_fs.get_sample_frequencies(),
        sim_data_fs,
        label="Interferometer FS",
    )
    axis[1].plot(
        psd.get_sample_frequencies(),
        psd,
        label="Noise PSD",
    )
    axis[1].set_title("Sim and Reconstructions FS")
    axis[1].legend(loc="upper right")

    axis[2].plot(tdi_data.sample_times, tdi_data)
    axis[2].set_title("TDI " + tdi_channel)

    plt.subplots_adjust(hspace=0.5)

    plt.savefig(PATH_tdi_backtracking_plots + plot_output)


def plot_psd(psd, name):
    figure, axis = plt.subplots(2, 1, figsize=(8, 8))

    axis[0].plot(
        psd.get_sample_frequencies(),
        psd,
        label="psd",
    )
    axis[0].legend(loc="upper right")

    plt.savefig(PATH_tdi_backtracking_plots + name)


def plot_level_performance(levels, identified, predicted, expected, num_ranges):
    min_level = min(levels)
    max_level = max(levels)
    step = int((max_level - min_level) / num_ranges)
    level_cutoffs = range(min_level, max_level, step)
    num_glitches = len(levels)

    analysis = {}

    for i in range(len(level_cutoffs) - 1):
        lower_bound = level_cutoffs[i]
        upper_bound = level_cutoffs[i + 1]

        analysis[str(lower_bound)] = {"identified": 0, "total": 0}

        for i in range(num_glitches):
            if levels[i] < upper_bound and levels[i] >= lower_bound:
                if identified[i] == "True":
                    analysis[str(lower_bound)]["identified"] += 1
                analysis[str(lower_bound)]["total"] += 1

    level_ranges = []
    success_rates = []
    totals = []

    for key, value in analysis.items():
        level_ranges.append(key)
        if value["total"] == 0:
            success_rates.append(0)
        else:
            success_rates.append(value["identified"] / value["total"])
        totals.append(value["total"])

    figure, axis = plt.subplots(2, 1, figsize=(12, 8))

    axis[0].bar(level_ranges, success_rates, align="edge", width=1)
    axis[0].set_title("Success Rates")

    axis[1].bar(level_ranges, totals, align="edge", width=1)
    axis[1].set_title("Totals")

    plt.savefig(PATH_tdi_backtracking_plots + "level_performance.png")


def print_success_rate(levels, identified, predicted, expected):
    num_success = 0
    num_glitches = len(levels)

    for i in range(num_glitches):
        if identified[i] == "False":
            print(f"{i} -- PREDICTED: {predicted[i]}, EXPECTED: {expected[i]}, LEVEL: {levels[i]}")
        else:
            num_success += 1

    print(f"{num_success}/{num_glitches} inj_points identified successfuly")


def print_interferometer_disribution(predicted, expected):
    distribution = {
        "isi": {"isi": 0, "tmi": 0, "rfi": 0},
        "tmi": {"isi": 0, "tmi": 0, "rfi": 0},
        "rfi": {"isi": 0, "tmi": 0, "rfi": 0},
    }

    for i in range(len(levels)):
        if predicted[i] != expected[i] and expected[i] != "gw":
            distribution[expected[i][:3]][predicted[i][:3]] += 1

    for key, value in distribution.items():
        print(f"{key}: {value}")


def examine_std(std, identified):
    identified_std = []
    unidentified_std = []

    for i in range(len(std)):
        if identified[i] == "True":
            identified_std.append(std[i])
        else:
            unidentified_std.append(std[i])

    print(f"Identified -- Mean: {np.mean(identified_std)} -- Min: {min(identified_std)} -- Max: {max(identified_std)} -- STD: {np.std(identified_std)}")
    print(f"Unidentified -- Mean: {np.mean(unidentified_std)} -- Min: {min(unidentified_std)} -- Max: {max(unidentified_std)} -- STD: {np.std(unidentified_std)}")


def plot_anomaly_outlier_distributions(glitches, gws):
    glitch_success_outliers = []
    glitch_unsuccess_outliers = []
    gw_outliers = []

    for glitch in glitches:
        if glitch["expected"] == glitch["predicted"]:
            glitch_success_outliers.append(max(glitch["outliers"].items(), key=lambda x: x[1])[1])
        else:
            glitch_unsuccess_outliers.append(max(glitch["outliers"].items(), key=lambda x: x[1])[1])

    for gw in gws:
        gw_outliers.append(max(gw["outliers"].items(), key=lambda x: x[1])[1])

    figure, axis = plt.subplots(1, 1, figsize=(8, 4))

    max_bin = int(max(glitch_success_outliers + glitch_unsuccess_outliers)) + 1
    step = 0.1
    bins = [step * i for i in range(int(max_bin / step))]
    # bins = [step * i for i in range(int(8 / step))]

    axis.hist(gw_outliers, bins=bins, label="GW")
    axis.hist(glitch_success_outliers, bins=bins, label="Glitch Success")
    axis.hist(glitch_unsuccess_outliers, bins=bins, label="Glitch Unsuccess")
    axis.axvline(GLITCH_CANDIDATE_THRESHOLD, ls="--", c="red")
    axis.legend(loc="upper right")
    axis.set_title("Anomaly Outlier Distribution")

    plt.savefig(PATH_tdi_backtracking_performance_plots + "anomaly_outlier_distrbution.png")


def init_anomalies(durations, levels, predicted, expected, outliers):
    anomalies = []
    gws = []
    glitches = []

    for i in range(len(levels)):
        outlier_dict = {}
        j = 0
        for interferometer in INTERFEROMETERS:
            for mosa in MOSAS:
                outlier_dict[f"{interferometer}_{mosa}"] = outliers[i][j]
                j += 1

        if max(outliers[i]) < GLITCH_CANDIDATE_THRESHOLD:
            suspect_gw = True
        else:
            suspect_gw = False

        anomaly = {
            "i": i,
            "duration": durations[i],
            "level": levels[i],
            "predicted": predicted[i],
            "expected": expected[i],
            "suspect_gw": suspect_gw,
            "outliers": outlier_dict,
        }

        anomalies.append(anomaly)

        if anomaly["expected"] == "gw":
            gws.append(anomaly)
        else:
            glitches.append(anomaly)

    return anomalies, gws, glitches


def filter_anomalies(anomalies, condition):
    subset = []
    for anomaly in anomalies:
        if condition(anomaly):
            subset.append(anomaly)
    return subset


def print_anomaly(anomaly):
    print(f"{anomaly['i']} -- SUSPECT_GW: {anomaly['suspect_gw']} -- PREDICTED: {anomaly['predicted']} -- EXPECTED: {anomaly['expected']} -- LEVEL: {anomaly['level']} -- DURATION: {anomaly['duration']} -- OUTLIER: {max(list(anomaly['outliers'].values()))}")


def print_performance(glitches, gws):
    print("~~~~~~~~~~ Anomaly Classification ~~~~~~~~~~")
    print("False Positives:")
    false_positives = filter_anomalies(
        anomalies=glitches,
        condition=lambda x: x["suspect_gw"]
    )
    for false_positive in false_positives:
        print_anomaly(false_positive)
    print(f"False positive rate: {len(false_positives)}/{len(glitches)}={len(false_positives)/len(glitches)}\n")

    print("False Negatives:")
    false_negatives = filter_anomalies(
        anomalies=gws,
        condition=lambda x: not x["suspect_gw"]
    )
    for false_negative in false_negatives:
        print_anomaly(false_negative)
    print(f"False negative rate: {len(false_negatives)}/{len(gws)}={len(false_negatives)/len(gws)}\n")

    print("Correctly Identified GW:")
    correct_gws = filter_anomalies(
        anomalies=gws,
        condition=lambda x: x["suspect_gw"]
    )
    print(f"Correctly identified gws: {len(correct_gws)}/{len(gws)}={len(correct_gws)/len(gws)}\n")

    print("~~~~~~~~~~ Glitch Identification ~~~~~~~~~~")
    print("Incorrectly Identified:")
    incorrect_glitches = filter_anomalies(
        anomalies=glitches,
        condition=lambda x: x["predicted"] != x["expected"] and x["expected"] != "gw"
    )
    for incorrect_glitch in incorrect_glitches:
        print_anomaly(incorrect_glitch)
    print(f"Successful glitch identification rate: {(len(glitches) - len(incorrect_glitches))}/{len(glitches)}={1 - len(incorrect_glitches)/len(glitches)}\n")

    print("Correctly Identified but False Positive:")
    correct_and_false_positives = filter_anomalies(
        anomalies=glitches,
        condition=lambda x: x["predicted"] == x["expected"] and x["suspect_gw"]
    )
    for correct_and_false_positive in correct_and_false_positives:
        print_anomaly(correct_and_false_positive)
    print(f"Correct identification but false positive rate: {len(correct_and_false_positives)}/{len(glitches)}={len(correct_and_false_positives)/len(glitches)}\n")



if __name__ == "__main__":
    num_seg = len(os.listdir(PATH_tdi_backtracking_results))

    if ".DS_Store" in os.listdir(PATH_tdi_backtracking_results):
        num_seg -= 1

    num_seg = 6

    durations = []
    levels = []
    identified = []
    predicted = []
    expected = []
    outliers = []

    fn = "results"

    for i in range(num_seg):
        glitch_info_str = np.genfromtxt(PATH_tdi_backtracking_results + f"{i}{fn}.txt", dtype=str)
        glitch_info_float = np.genfromtxt(PATH_tdi_backtracking_results + f"{i}{fn}.txt", dtype=float)

        durations += map(int, list(glitch_info_float[0:, 1]))
        levels += map(int, list(glitch_info_float[0:, 2]))
        predicted += list(glitch_info_str[0:, 3])
        expected += list(glitch_info_str[0:, 4])
        outliers += list(glitch_info_float[0:, 5:])

    anomalies, gws, glitches = init_anomalies(
        durations=durations,
        levels=levels,
        predicted=predicted,
        expected=expected,
        outliers=outliers,
    )

    print_performance(
        glitches=glitches,
        gws=gws,
    )

    # plot_level_performance(
    #     levels=levels,
    #     identified=identified,
    #     predicted=predicted,
    #     expected=expected,
    #     num_ranges=100,
    # )

    # print_success_rate(
    #     levels=levels,
    #     identified=identified,
    #     predicted=predicted,
    #     expected=expected,
    # )

    # print_interferometer_disribution(
    #     predicted=predicted,
    #     expected=expected,
    # )

    plot_anomaly_outlier_distributions(
        glitches=glitches,
        gws=gws,
    )
