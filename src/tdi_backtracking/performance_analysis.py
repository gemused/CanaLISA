import os
import matplotlib.pyplot as plt
import numpy as np

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_tdi_backtracking_plots = os.path.join(PATH_bethLISA,
                                           "dist/tdi_backtracking_plots/")
PATH_tdi_backtracking_results = os.path.join(PATH_bethLISA,
                                             "dist/tdi_backtracking_results/")
GLITCH_CANDIDATE_THRESHOLD = 5
OUTLIERS_THRESHOLD_MAX = 1
OUTLIERS_THRESHOLD_MIN = 2

def plot_overlap(
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

    plt.savefig(PATH_tdi_backtracking_plots + "performance.png")


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
        if predicted[i] != expected[i] and predicted[i] != "gw":
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


if __name__ == "__main__":
    num_seg = len(os.listdir(PATH_tdi_backtracking_results))

    if ".DS_Store" in os.listdir(PATH_tdi_backtracking_results):
        num_seg -= 1

    levels = []
    identified = []
    predicted = []
    expected = []
    outliers = []

    for i in range(num_seg):
        glitch_info_str = np.genfromtxt(PATH_tdi_backtracking_results + f"{i}results.txt", dtype=str)
        glitch_info_float = np.genfromtxt(PATH_tdi_backtracking_results + f"{i}results.txt", dtype=float)

        levels += map(int, list(glitch_info_float[0:, 1]))
        identified += list(glitch_info_str[0:, 2])
        predicted += list(glitch_info_str[0:, 3])
        expected += list(glitch_info_str[0:, 4])
        outliers += list(glitch_info_float[0:, 5:])

    plot_level_performance(
        levels=levels,
        identified=identified,
        predicted=predicted,
        expected=expected,
        num_ranges=20,
    )

    print_success_rate(
        levels=levels,
        identified=identified,
        predicted=predicted,
        expected=expected,
    )

    print_interferometer_disribution(
        predicted=predicted,
        expected=expected,
    )
