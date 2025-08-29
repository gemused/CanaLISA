
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

from pathlib import Path

from scipy.optimize import fsolve
from scipy.signal import savgol_filter
from scipy.ndimage import label
from scipy.signal.windows import tukey
from scipy.integrate import simpson

from numpy.lib.stride_tricks import sliding_window_view

cwd = Path().cwd()
PATH_bethLISA = cwd.parents[1]
PATH_tdi_data = PATH_bethLISA / "dist/lisa_data/tdi_data"
PATH_glitch_data = PATH_bethLISA / "dist/glitch/glitch_data"
PATH_gw_data = PATH_bethLISA / "dist/gw/gw_data"
PATH_gating_data = PATH_bethLISA / "dist/canalysis"


tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

downsample = 4
sampling_rate = 4 // downsample       
window_seconds = 120 
window_length = sampling_rate * window_seconds  
step_seconds = 30
step_size = sampling_rate * step_seconds 
tdi_folder = "1d50apdLoud"  
variables = "XYZ"

"""
Set tdi_folder above to the name of the data you want to analyze and gate. Set variables to the TDI channels you want to use. 
"XYZ" or "AET" available. This doesn't really do anything except load the data differently, use a different model trained on those 
channels, and return the gated data with these same channels.
"""


def load_time_series_data(folder, variables):
    """
    Loads TDI data, fetches glitch and gw injection times and loudness from h5 files.

    Arguments:
        folder: name of TDI data folder. Data in any h5 file in any folder within this folder will be concatenated. I name these subfolders 
                with integers ascending from 1 when I have multiple simulations I want to join into one bigger one.

    Returns:
        data: TDI data of shape (seconds, 3)
    """

    tdi1, tdi2, tdi3 = [], [], []
    
    #READ TDI DATA TO SINGLE ARRAY FROM FILES
    for folder_path in sorted((f for f in os.listdir(os.path.join(PATH_tdi_data, folder)) if not f.startswith("."))):
        files = sorted((f for f in os.listdir(os.path.join(PATH_tdi_data, folder, folder_path)) if not f.startswith(".")))
        for fname in files:
            path = os.path.join(PATH_tdi_data, folder, folder_path, fname)
            with h5py.File(path, 'r') as hf:
                X = (hf['X'][:][::downsample])
                Y = (hf['Y'][:][::downsample])
                Z = (hf['Z'][:][::downsample])
                if variables == 'XYZ':
                    tdi1.append(X); tdi2.append(Y); tdi3.append(Z) 
                elif variables == 'AET':
                    tdi1.append((Z - X) / 2 ** 0.5)
                    tdi2.append((X - 2*Y + Z) / 6**0.5)
                    tdi3.append((X + Y + Z) / 3**0.5)

    data = (np.stack([np.concatenate(tdi1),np.concatenate(tdi2),np.concatenate(tdi3)], axis=-1))

    return data

test_data = load_time_series_data(folder=tdi_folder, variables=variables)
if variables == "AET":
    model = tf.keras.models.load_model('CanalisaAET.keras')
elif variables == "XYZ":
    model = tf.keras.models.load_model('CanalisaXYZ.keras')
    
def test_dataset_from_data(data, window_length, step_size):
    """
    Slices test data into windows to run inference on. Similar to train_dataset_from_data except labels are not needed.

    Arguments:
        data: test data. shape (seconds, 3)
        window_length: length of windows in seconds.
        step_size: step between start times of windows in seconds.

    Returns:
        test_ds: tf.data.Dataset object. Unshuffled, unlabelled windows of TDI data to perform inference on.
    """
    test_wins = sliding_window_view(data, window_length, axis=0)[::step_size].transpose(0,2,1)
    test_ds = (tf.data.Dataset.from_tensor_slices(test_wins).batch(8).prefetch(tf.data.AUTOTUNE))
    return test_ds

def accumulate_predictions(data_len, preds, step_size, window_length):
    """
    Pieces together the predicted masks from each window after running inference on them.

    Arguments:
        data_len: total length of the data in seconds.
        preds: Output from model.predict() on the dataset. I.e. the model's predicted masks.
        step_size: step between start times of windows in seconds.
        window_length: window length in seconds.

    Returns:
        scores: The accumulated predictions. Shape (seconds, 2). Note these are affected by repeated counts of overlapping windows.
        counts: Shape (seconds, 2). Number of times each sample has been counted due to overlapping windows. Used to get correct predictions from scores.
    """
    scores = np.zeros((data_len, 2), dtype=np.float32)
    counts = np.zeros((data_len, 2), dtype=np.int32)
    #LOOP OVER WINDOWS, i IS WINDOW NUMBER, PROB HAS SHAPE (window_length, 2)
    for i, prob in enumerate(preds):
        start = i * step_size
        #ADD mask FROM WINDOW TO SCORE ARRAY
        scores[start:start+window_length] += prob
        #COUNT HOW MANY TIMES THIS WINDOW AS BEEN COVERED TO COMPUTE CORRECT AVERAGES
        counts[start:start+window_length] += 1
    return scores, counts

def inference(data, smoothing):
    """
    Generates a predicted GW and glitch mask for a set of test data.

    Arguments:
        data: TDI data we want to do predictions for.
        smoothing: Integer > 0, determines how smooth the mask is. Larger number = more smoothing. I tend to use 50 to 100.
    
    Returns: 
        glitch_mask: Shape (seconds). Array between 0 and 1. Represents probability of glitch at any given second.
        gw_mask: Shape (seconds). Array between 0 and 1. Represents probability of GW at any given second.
    """
    dataset = test_dataset_from_data(data=data, window_length=window_length, step_size=step_size)

    preds = model.predict(dataset, verbose=1)

    scores, counts = accumulate_predictions(len(data), preds, step_size, window_length)

    average_mask = scores / np.maximum(counts, 1)

    glitch_mask = savgol_filter(average_mask[:,0], smoothing, 4)
    gw_mask = savgol_filter(average_mask[:,1], smoothing, 4)

    return glitch_mask, gw_mask

glitch_mask, gw_mask = inference(data=test_data, smoothing=100)

def extract_events(mask, small_len, big_len, log_threshold, eps = 1e-9, merge_adjacent = True):
    """
    Give scores between -inf and 0 to windows of data based on likelihood of anomaly from probability mask and return times and confidence
    of predicted anomalies from this.

    Arguments:
        mask: probability mask (from inference())
        small_len: stride (or size, since there is no overlap) of small windows over the mask, calculating 
                   the mean of the window.
        big_len: stride of big windows over the small windows. Calculates the sum of the log of each small window.

    Returns:
        segments: list of dicts with start_idx, end_idx (inclusive, in samples/seconds),
                  start_small, end_small (indices in small-windows),
                  start_big, end_big (indices in big-windows),
                  and scores for each big window included.
        times: list of (start, end) tuples of windows containing predicted anomalies. 
        confidences: scores of predicted windows. If multiple passing windows in succession, average score is given.
    """

    mask = np.asarray(mask, dtype=float)
    N = mask.shape[0]

    # MEANS OVER SMALL WINDOWS
    n_small = N // small_len
    if n_small == 0:
        return []
    trimmed = mask[: n_small * small_len]
    small_windows = trimmed.reshape(n_small, small_len)
    small_means = small_windows.mean(axis=1)  

    # LOGS OVER BIG WINDOWS
    n_big = n_small // big_len
    if n_big == 0:
        return []
    trimmed_small = small_means[: n_big * big_len]
    big_windows = trimmed_small.reshape(n_big, big_len)
    scores = np.log(np.clip(big_windows, eps, 1.0)).sum(axis=1)  # CLIP TO AVOID INFINITY

    # THRESHOLD
    passing = scores >= log_threshold  

    segments = []
    times = []
    confidences = []

    i = 0
    while i < n_big:
        if not passing[i]:
            i += 1
            continue

        
        j = i
        while j + 1 < n_big and passing[j + 1] and merge_adjacent:
            j += 1

        # MAP WINDOW INDICES BACK TO SAMPLE INDICES TO GET THE CORRECT TIMES
        start_small = i * big_len
        end_small = (j + 1) * big_len - 1  

        start_idx = start_small * small_len
        end_idx = (end_small + 1) * small_len - 1 

        seg_scores = scores[i : j + 1].copy()

        segments.append({
            "start_idx": int(start_idx),
            "end_idx":   int(end_idx),
            "start_small": int(start_small),
            "end_small":   int(end_small),
            "start_big":   int(i),
            "end_big":     int(j),
            "big_window_scores": seg_scores,
            "min_score": float(seg_scores.min()),
            "max_score": float(seg_scores.max()),
            "mean_score": float(seg_scores.mean()),
        })

        times.append((int(start_idx), int(end_idx)))
        confidences.append(seg_scores.mean())
        

        i = j + 1

    return times, confidences

glitch_events, glitch_scores = extract_events(mask=glitch_mask, small_len=2, big_len=16, log_threshold=-4)
gw_events, gw_scores = extract_events(mask=gw_mask, small_len=2, big_len=19, log_threshold=-7)

def gate(data, glitch_where, gw_where, roundedness):
    """
    Tapers regions within and around detected anomalies to zero.

    Arguments:
        data: TDI data to gate
        glitch_where, gw_where: arrays of (start, end) tuples of predicted glitch and gw detection times for the corresponding TDI data.
        roundedness: how steep the tukey window is. Using 0.5 generally.

    Returns:
        gated_data: TDI data with anomalous regions set to zero.
    """

    where = glitch_where + gw_where
    gated_data = data.copy()
    for start, end in where:
        npoints = (240+end-start)
        window = tukey(npoints, alpha=roundedness)
        try:
            for i in range(3): gated_data[start-120:end+120, i] *= (1-window)
        except ValueError:
            pass

    return gated_data

gated_data = gate(data=test_data, glitch_where=glitch_events, gw_where=gw_events, roundedness=0.5)

def output(gated_data, glitch_events, glitch_scores, gw_events, gw_scores, folder, variables):
    with open(os.path.join(PATH_gating_data, "glitch_output", folder+".txt"), "w") as f:
        f.write(f'start | end | score\n')
        for i in range(len(glitch_events)):
            f.write(f'{glitch_events[i][0]} {glitch_events[i][1]} {glitch_scores[i]}\n')

    with open(os.path.join(PATH_gating_data, "gw_output", folder+".txt"), "w") as f:
        f.write(f'start | end | score\n')
        for i in range(len(gw_events)):
            f.write(f'{gw_events[i][0]} {gw_events[i][1]} {gw_scores[i]}\n')

    with h5py.File(os.path.join(PATH_gating_data, "gated_data", folder+"_gated.h5"), 'w') as f: 
        if variables == "XYZ":
            f.create_dataset("X", data=gated_data[:,0])
            f.create_dataset("Y", data=gated_data[:,1])
            f.create_dataset("Z", data=gated_data[:,2])
        elif variables == "AET":
            f.create_dataset("A", data=gated_data[:,0])
            f.create_dataset("E", data=gated_data[:,1])
            f.create_dataset("T", data=gated_data[:,2])

output(gated_data=gated_data, glitch_events=glitch_events, glitch_scores=glitch_scores, gw_events=gw_events, gw_scores=gw_scores, folder=tdi_folder, variables=variables)       


