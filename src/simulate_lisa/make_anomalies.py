"""
Filename: make_anomalies.py
Author: William Mostrenko
Created: 2025-07-18
Description: Creates glitches and gravitational waves from parameters specified
in config files.
"""

import os
import sys
import lisaglitch
import numpy as np
import ldc.io.yml as ymlio
import argparse
from lisagwresponse import ResponseFromStrain
from lisaglitch import OneSidedDoubleExpGlitch, TwoSidedDoubleExpGlitch, StepGlitch
from scipy.optimize import fsolve

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_pipe_config = os.path.join(PATH_bethLISA, "dist/pipe/pipe_config/")
PATH_glitch_config = os.path.join(PATH_bethLISA, "dist/glitch/glitch_config/")
PATH_gw_config = os.path.join(PATH_bethLISA, "dist/gw/gw_config/")
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch/glitch_data/")
PATH_gw_data = os.path.join(PATH_bethLISA, "dist/gw/gw_data/")
PATH_pipe_data = os.path.join(PATH_bethLISA, "dist/pipe/pipe_data/")
PATH_orbits_data = os.path.join(PATH_bethLISA, "dist/lisa_data/orbits_data/")
PATH_anomaly_data = os.path.join(PATH_bethLISA, "dist/tdi_backtracking/anomaly_data/")
t0 = 10368000


class GWFRED(ResponseFromStrain):
    """Represents a one-sided double-exponential gw signal

    Args:
        t_rise: Rising timescale
        t_fall: Falling timescale
        level: amplitude
    """
    def __init__(
        self,
        t_rise: float,
        t_fall: float,
        level: float,
        t_inj: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.t_rise = float(t_rise)
        self.t_fall = float(t_fall)
        self.level = float(level)
        self.t_inj = float(t_inj)

    def compute_hcross(self, t):
        return self.compute_signal(t)

    def compute_hplus(self, t):
        return self.compute_signal(t)

    def compute_signal(self, t):
        """Computes the FRED response model.

        Args:
            t (array-like): Times to compute GW model for.

        Returns:
            Computed FRED model (array-like)
        """
        offset = 20
        delta_t = t - self.t_inj + offset + (8.5 / 86400) * (self.t_inj - t0)

        if self.t_rise != self.t_fall:
            exp_terms = np.exp(-delta_t / self.t_rise) - np.exp(-delta_t / self.t_fall)
            signal = self.level * exp_terms / (self.t_rise - self.t_fall)
        else:
            signal = self.level * delta_t * np.exp(-delta_t / self.t_fall) / self.t_fall**2

        return np.where(delta_t >= 0, signal, 0)


def make_t_injs(glitch_rate, gw_rate, duration, window):
    """Makes a list of anomaly injection times.

    Args:
        glitch_rate (int): Number of glitches per day.
        gw_rate (int): Number of gravitational wave signals per day.
        duration (float): Simulation duration in seconds.
        window (float): Minimum distance between injection times in seconds.

    Returns:
        Anomaly injection times (list)
    """
    t_injs = []

    for i in range(int((glitch_rate + gw_rate) * duration / 86400)):
        invalid_t_inj = True
        while invalid_t_inj:
            t_inj = np.random.randint(t0, t0 + duration)
            if t_injs:
                for t in t_injs:
                    if t_inj < t + window and t_inj > t - window and t_inj + window < t0 + duration:
                        invalid_t_inj = True
                        break
                    else:
                        invalid_t_inj = False
            else:
                invalid_t_inj = False
        t_injs.append(t_inj)

    return t_injs


def compute_anomalies_params(
    glitch_cfg, gw_cfg, pipe_cfg, orbits_input_fn, t_injs
):
    """Compute dictionaries of parameters for each glitch and gravitational wave
    signal to be injected.

    Args:
        glitch_cfg (dict): Glitch configuration dict containing glitch and
            injection parameters.
        gw_rate (dict): Gravitational wave configuration dict containing gw
            and injection parameters.
        pipe_cfg (dict): Pipeline configuration dict containing lisa instrument
            simulation parameters.
        orbits_input_fn (str): Orbits h5 data file name (excluding file extensions).
        t_injs (array-like): List of anomaly injection times.

    Returns:
        List of dictionary of glitch parameters, list of dictionary of gw parameters
    """
    glitches_params = []
    gws_params = []

    anomaly_rate = glitch_cfg["daily_rate"] + gw_cfg["daily_rate"]

    for t_inj in t_injs:
        if np.random.choice((True, False), p=(glitch_cfg["daily_rate"] / anomaly_rate, gw_cfg["daily_rate"] / anomaly_rate)):
            if glitch_cfg["shape"] == "OneSidedDoubleExpGlitch":
                t_fall_range = glitch_cfg["t_fall_range"]
                amp_range = glitch_cfg["amp_range"]

                amp = np.random.uniform(float(amp_range[0]), float(amp_range[1]))
                t_fall = np.random.randint(t_fall_range[0], t_fall_range[1])
                level = amp * t_fall

                glitches_params.append(
                    {
                        "shape": "OneSidedDoubleExpGlitch",
                        "inj_point": np.random.choice(glitch_cfg["inj_points"]),
                        "t_rise": t_fall,
                        "t_fall": t_fall,
                        "level": level,
                        "t_inj": t_inj,
                    }
                )
            elif glitch_cfg["shape"] == "TwoSidedDoubleExpGlitch":
                t_rise_range = glitch_cfg["t_rise_range"]
                t_fall_range = glitch_cfg["t_fall_range"]
                level_range = glitch_cfg["level_range"]
                displacement_range = glitch_cfg["displacement_range"]

                glitches_params.append(
                    {
                        "shape": "TwoSidedDoubleExpGlitch",
                        "inj_point": np.random.choice(glitch_cfg["inj_points"]),
                        "t_rise": np.random.randint(t_rise_range[0], t_rise_range[1]),
                        "t_fall": np.random.randint(t_fall_range[0], t_fall_range[1]),
                        "level": np.random.uniform(float(level_range[0]), float(level_range[1])),
                        "displacement": np.random.uniform(float(displacement_range[0]), float(displacement_range[1])),
                        "t_inj": t_inj,
                    }
                )
            elif glitch_cfg["shape"] == "StepGlitch":
                level_range = glitch_cfg["level_range"]

                glitches_params.append(
                    {
                        "shape": "StepGlitch",
                        "inj_point": np.random.choice(glitch_cfg["inj_points"]),
                        "level": np.random.uniform(float(level_range[0]), float(level_range[1])),
                        "t_inj": t_inj,
                    }
                )
            else:
                raise AttributeError("Unsupported glitch shape")
        else:
            if gw_cfg["shape"] == "GWFRED":
                t_fall_range = gw_cfg["t_fall_range"]
                amp_range = gw_cfg["amp_range"]

                amp = np.random.uniform(float(amp_range[0]), float(amp_range[1]))
                t_fall = np.random.randint(t_fall_range[0], t_fall_range[1])
                level = amp * t_fall

                gws_params.append(
                    {
                        "shape": "GWFRED",
                        "t_rise": t_fall,
                        "t_fall": t_fall,
                        "level": level,
                        "t_inj": t_inj,
                    }
                )
            else:
                raise AttributeError("Unsupported gravitational wave shape")

    return glitches_params, gws_params


def compute_glitches(glitches_params, dt, size):
    """Create glitch objects.

    Args:
        glitches_params (list<dict>): List of glitch parameter dictionaries.
        dt (float): Sampling time steps for LISA instrument.
        size (float): Number of output data points in lisa simulation.

    Returns:
        List of glitch objects
    """
    glitches = []

    for glitch_params in glitches_params:
        if glitch_params["shape"] == "OneSidedDoubleExpGlitch":
            glitches.append(
                OneSidedDoubleExpGlitch(
                    inj_point=glitch_params["inj_point"],
                    t_inj=glitch_params["t_inj"],
                    t_rise=glitch_params["t_fall"],
                    t_fall=glitch_params["t_fall"],
                    level=glitch_params["level"],
                    t0=t0,
                    size=size,
                    dt=dt,
                )
            )
        elif glitch_params["shape"] == "TwoSidedDoubleExpGlitch":
            glitches.append(
                TwoSidedDoubleExpGlitch(
                    inj_point=glitch_params["inj_point"],
                    t_inj=glitch_params["t_inj"],
                    t_rise=glitch_params["t_rise"],
                    t_fall=glitch_params["t_fall"],
                    level=glitch_params["level"],
                    displacement=glitch_params["displacement"],
                    t0=t0,
                    size=size,
                    dt=dt,
                )
            )
        elif glitch_params["shape"] == "StepGlitch":
            glitches.append(
                StepGlitch(
                    inj_point=glitch_params["inj_point"],
                    t_inj=glitch_params["t_inj"],
                    level=glitch_params["level"],
                    t0=t0,
                    size=size,
                    dt=dt,
                )
            )
        else:
            raise AttributeError("Unsupported glitch shape")

    return glitches


def compute_gws(gws_params, orbits_input_fn, dt, size):
    """Create gravitational wave objects.

    Args:
        gws_params (list<dict>): List of gravitational wave parameter dictionaries.
        dt (float): Sampling time steps for LISA instrument.
        size (float): Number of output data points in lisa simulation.

    Returns:
        List of gravitational wave objects
    """
    gws = []

    for gw_params in gws_params:
        if gw_params["shape"] == "GWFRED":
            gw_beta = 0#np.random.uniform(-np.pi/2, np.pi/2)
            gw_lambda = np.pi/7#np.random.uniform(-np.pi/2, np.pi/2)

            gws.append(
                GWFRED(
                    t_inj=gw_params["t_inj"],
                    t_rise=gw_params["t_fall"],
                    t_fall=gw_params["t_fall"],
                    level=gw_params["level"],
                    gw_beta=gw_beta,
                    gw_lambda=gw_lambda,
                    orbits=PATH_orbits_data + orbits_input_fn + ".h5",
                    dt=dt,
                    size=size,
                    t0=t0,
                )
            )
        else:
            raise AttributeError("Unsupported gravitational wave shape")

    return gws


def write(
    glitches, gws, glitch_output_path, gws_output_path, pipe_output_path,
    anomaly_output_path, t0, dt, size
):
    """Write anomalies to respective h5 files and their relevant information
    txt files.

    Args:
        glitches (list): List of lisaglitch glitch objects.
        gws (list<dict>): List of gwresponse gravitational wave objects.
        glitch_output_path (str): Path to write glitch data and info to (excluding file extension).
        gws_output_path (str): Path to write gravitational wave data and info to (excluding file extension).
        pipe_output_path (str): Path to write LISA pipeline info to (excluding file extension).
        anomaly_output_path (str): Path to write anomaly info to (excluding file extension).
        t0 (float): Initial time of simulation.
        dt (float): Sampling time steps for LISA instrument.
        size (float): Number of output data points in lisa simulation.

    Returns:
        None
    """
    glitch_output_path_h5 = glitch_output_path + ".h5"
    glitch_output_path_txt = glitch_output_path + ".txt"

    gws_output_path_h5 = gws_output_path + ".h5"
    gws_output_path_txt = gws_output_path + ".txt"

    pipe_output_path_txt = pipe_output_path + ".txt"

    anomaly_output_path_txt = anomaly_output_path + ".txt"

    for path in [glitch_output_path_h5, glitch_output_path_txt, gws_output_path_h5, gws_output_path_txt, pipe_output_path_txt, anomaly_output_path_txt]:
        if os.path.exists(path):
            os.remove(path)

    for glitch in glitches:
        glitch.write(path=glitch_output_path_h5, mode="a")

    for gw in gws:
        gw.write(path=gws_output_path_h5, mode="a")

    with open(f"{glitch_output_path_txt}", "w") as f:
        for glitch in glitches:
            if isinstance(glitch, OneSidedDoubleExpGlitch):
                duration, amp = osde_properties(glitch)
                f.write(f"OneSidedDoubleExpGlitch {glitch.t_inj} {amp} {glitch.t_rise} {glitch.t_fall} {duration}\n")
            elif isinstance(glitch, TwoSidedDoubleExpGlitch):
                f.write(f"TwiSidedDoubleExpGlitch {glitch.t_inj} {glitch.level} {glitch.t_rise} {glitch.t_fall} {glitch.displacement}\n") # NEED TO ADD DURATION
            elif isinstance(glitch, StepGlitch):
                f.write(f"StepGlitch {glitch.t_inj} {glitch.level}\n") # NEED TO ADD DURATION
            else:
                raise AttributeError("Unsupported glitch shape")

    with open(f"{gws_output_path_txt}", "w") as f:
        for gw in gws:
            if isinstance(gw, GWFRED):
                duration, amp = osde_properties(gw)
                f.write(f"GWFRED {gw.t_inj} {amp} {gw.t_rise} {gw.t_fall} {duration}\n")
            else:
                raise AttributeError("Unsupported gravitational wave shape")

    with open(f"{pipe_output_path_txt}", "w") as f:
        f.write(f"{t0} {dt} {size}")

    with open(f"{anomaly_output_path_txt}", "w") as f:
        anomalies = glitches + gws
        anomalies.sort(key=lambda x: x.t_inj)
        for anomaly in anomalies:
            if isinstance(anomaly, OneSidedDoubleExpGlitch):
                duration, amp = osde_properties(anomaly)
                f.write(f"OneSidedDoubleExpGlitch {anomaly.inj_point} {anomaly.t_inj} {amp} {duration}\n")
            elif isinstance(anomaly, TwoSidedDoubleExpGlitch):
                f.write(f"TwoSidedDoubleExpGlitch {anomaly.inj_point} {anomaly.t_inj} {anomaly.level}\n") # NEED TO ADD DURATION
            elif isinstance(anomaly, StepGlitch):
                f.write(f"StepGlitch {anomaly.inj_point} {anomaly.t_inj} {anomaly.level}\n") # NEED TO ADD DURATION
            elif isinstance(anomaly, GWFRED):
                duration, amp = osde_properties(anomaly)
                f.write(f"GWBurst gw {anomaly.t_inj} {amp} {duration}\n")
            else:
                raise AttributeError("Unsupported anomaly shape")


def osde_properties(anomaly):
    t_fall = anomaly.t_fall
    level = anomaly.level

    amp = level / (t_fall * np.exp(1))
    t_max = t_fall
    guess = t_max + 50

    roots = lambda t: g_FRED(t) - amp / 30
    g_FRED = lambda t: (level * (t / (t_fall ** 2)) * np.exp(-t / t_fall))

    duration = int(fsolve(roots, guess)[0] + 5*8)

    return duration, amp


def make_anomalies(
    glitch_cfg_input_fn, gw_cfg_input_fn, pipe_cfg_input_fn, orbits_input_fn,
    glitch_output_fn, gw_output_fn, pipe_output_fn, anomaly_output_fn
):
    """Creates glitch and gravitational wave objects given config files and
    writes data to their respective files.

    Args:
        glitch_cfg_input_fn (str): Glitch config file name (excluding file extension).
        gw_cfg_input_fn (str): Gravitational wave config file name (excluding file extension).
        pipe_cfg_input_fn (str): Pipeline config file name (excluding file extension).
        orbits_input_fn (str): Orbits h5 data file name (excluding file extensions).
        glitch_output_fn (str): Glitch h5/txt data output file name (excluding file extensions).
        gw_output_fn (str): GW h5/txt data output file name (excluding file extensions).
        pipe_output_fn (str): Pipeline txt data output file name (excluding file extensions).
        anomaly_output_fn (str): Anomaly txt data output file name (excluding file extensions). 

    Returns:
        None
    """
    glitch_cfg = ymlio.load_config(PATH_glitch_config + glitch_cfg_input_fn + ".yml")
    gw_cfg = ymlio.load_config(PATH_gw_config + gw_cfg_input_fn + ".yml")
    pipe_cfg = ymlio.load_config(PATH_pipe_config + pipe_cfg_input_fn + ".yml")

    dt = pipe_cfg["dt"].to("s").value
    size = pipe_cfg["duration"].to("s").value / dt

    if "glitch_0" in glitch_cfg and "gw_0" in gw_cfg:
        glitches_params = [value for key, value in glitch_cfg.items()]
        gws_params = [value for key, value in gw_cfg.items()]

        for glitch_params in glitches_params:
            glitch_params["t_inj"] += t0

        for gw_params in gws_params:
            gw_params["t_inj"] += t0
    elif "glitch_0" not in glitch_cfg and "gw_0" not in gw_cfg:
        t_injs = make_t_injs(
            glitch_rate=glitch_cfg["daily_rate"],
            gw_rate=gw_cfg["daily_rate"],
            duration=pipe_cfg["duration"].to("s").value,
            window=800,
        )

        glitches_params, gws_params = compute_anomalies_params(
            glitch_cfg=glitch_cfg,
            gw_cfg=gw_cfg,
            pipe_cfg=pipe_cfg,
            orbits_input_fn=orbits_input_fn,
            t_injs=t_injs,
        )
    else:
        raise AttributeError("Mix-and-matching single and large-scale gw and glitch injections unsupported")

    glitches = compute_glitches(
        glitches_params=glitches_params,
        dt=dt,
        size=size,
    )

    gws = compute_gws(
        gws_params=gws_params,
        orbits_input_fn=orbits_input_fn,
        dt=dt,
        size=size,
    )

    write(
        glitches=glitches,
        gws=gws,
        glitch_output_path=PATH_glitch_data + glitch_output_fn,
        gws_output_path=PATH_gw_data + gw_output_fn,
        pipe_output_path=PATH_pipe_data + pipe_output_fn,
        anomaly_output_path=PATH_anomaly_data + anomaly_output_fn,
        t0=t0,
        dt=dt,
        size=size,
    )
