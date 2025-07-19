import os
import sys
import lisaglitch
import numpy as np
import ldc.io.yml as ymlio
import argparse
from lisagwresponse import ResponseFromStrain
from lisaglitch import OneSidedDoubleExpGlitch


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
        delta_t = t - self.t_inj

        if self.t_rise != self.t_fall:
            exp_terms = np.exp(-delta_t / self.t_rise) - np.exp(-delta_t / self.t_fall)
            signal = self.level * exp_terms / (self.t_rise - self.t_fall)
        else:
            signal = self.level * delta_t * np.exp(-delta_t / self.t_fall) / self.t_fall**2

        return np.where(delta_t >= 0, signal, 0)


def make_t_injs(glitch_rate, gw_rate, duration, window):
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


def make_glitch(glitch_cfg, pipe_cfg, t_inj, dt, size):
    if glitch_cfg["shape"] == "OneSidedDoubleExpGlitch":
        t_rise_range = glitch_cfg["t_rise_range"]
        t_fall_range = glitch_cfg["t_fall_range"]
        level_range = glitch_cfg["level_range"]

        return OneSidedDoubleExpGlitch(
            inj_point=np.random.choice(glitch_cfg["inj_points"]),
            t0=t0,
            size=size,
            dt=dt,
            t_inj=t_inj,
            t_rise=np.random.randint(t_rise_range[0], t_rise_range[1]),
            t_fall=np.random.randint(t_fall_range[0], t_fall_range[1]),
            level=np.random.uniform(float(level_range[0]), float(level_range[1])),
        )
    else:
        raise AttributeError("Unsupported glitch shape")


def make_gw(gw_cfg, pipe_cfg, orbits_input_path, t_inj, dt, size):
    if gw_cfg["shape"] == "GWFRED":
        t_rise_range = gw_cfg["t_rise_range"]
        t_fall_range = gw_cfg["t_fall_range"]
        level_range = gw_cfg["level_range"]
        gw_beta = np.random.uniform(-np.pi/2, np.pi/2)
        gw_lambda = np.random.uniform(-np.pi/2, np.pi/2)

        return GWFRED(
            t_rise=np.random.randint(t_rise_range[0], t_rise_range[1]),
            t_fall=np.random.randint(t_fall_range[0], t_fall_range[1]),
            level=np.random.uniform(float(level_range[0]), float(level_range[1])),
            t_inj=t_inj,
            gw_beta=gw_beta,
            gw_lambda=gw_lambda,
            orbits=PATH_orbits_data + orbits_input_path + ".h5",
            dt=dt,
            size=size,
            t0=t0,
        )
    else:
        raise AttributeError("Unsupported gravitational wave shape")


def compute_anomalies(
    glitch_cfg, gw_cfg, pipe_cfg, orbits_input_path, t_injs, dt, size
):
    glitches = []
    gws = []

    anomaly_rate = glitch_cfg["daily_rate"] + gw_cfg["daily_rate"]

    for t_inj in t_injs:
        if np.random.choice((True, False), p=(glitch_cfg["daily_rate"] / anomaly_rate, gw_cfg["daily_rate"] / anomaly_rate)):
            glitches.append(
                make_glitch(
                    glitch_cfg=glitch_cfg,
                    pipe_cfg=pipe_cfg,
                    t_inj=t_inj,
                    dt=dt,
                    size=size,
                )
            )
        else:
            gws.append(
                make_gw(
                    gw_cfg=gw_cfg,
                    pipe_cfg=pipe_cfg,
                    orbits_input_path=orbits_input_path,
                    t_inj=t_inj,
                    dt=dt,
                    size=size,
                )
            )

    return glitches, gws


def write(
    glitches, gws, glitch_output_path, gws_output_path, pipe_output_path,
    anomaly_output_path, t0, dt, size
):
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
                f.write(f"OneSidedDoubleExpGlitch {glitch.t_inj} {glitch.level} {glitch.t_rise} {glitch.t_fall}\n")
            else:
                raise AttributeError("Unsupported glitch shape")

    with open(f"{gws_output_path_txt}", "w") as f:
        for gw in gws:
            if isinstance(gw, GWFRED):
                f.write(f"GWFRED {gw.t_inj} {gw.level} {gw.t_rise} {gw.t_fall}\n")
            else:
                raise AttributeError("Unsupported gravitational wave shape")

    with open(f"{pipe_output_path_txt}", "w") as f:
        f.write(f"{t0} {dt} {size}")

    with open(f"{anomaly_output_path_txt}", "w") as f:
        anomalies = glitches + gws
        anomalies.sort(key=lambda x: x.t_inj)
        for anomaly in anomalies:
            if isinstance(anomaly, OneSidedDoubleExpGlitch):
                f.write(f"OneSidedDoubleExpGlitch {anomaly.inj_point} {anomaly.t_inj} {anomaly.level}\n")
            elif isinstance(anomaly, GWFRED):
                f.write(f"GWBurst gw {anomaly.t_inj} {anomaly.level}\n")
            else:
                raise AttributeError("Unsupported anomaly shape")


def make_anomalies(
    glitch_cfg_input_fn, gw_cfg_input_fn, pipe_cfg_input_fn, orbits_input_fn,
    glitch_output_fn, gw_output_fn, pipe_output_fn, anomaly_output_fn
):
    glitch_cfg = ymlio.load_config(PATH_glitch_config + glitch_cfg_input_fn + ".yml")
    gw_cfg = ymlio.load_config(PATH_gw_config + gw_cfg_input_fn + ".yml")
    pipe_cfg = ymlio.load_config(PATH_pipe_config + pipe_cfg_input_fn + ".yml")

    dt = pipe_cfg["dt"].to("s").value
    size = pipe_cfg["duration"].to("s").value / dt

    if "glitch_0" in glitch_cfg and "gw_0" in gw_cfg:
        glitches = [value for key, value in glitch_cfg.items()]
        gws = [value for key, value in gw_cfg.items()]

        for glitch in glitches:
            glitch["t_inj"] += t0
        for gw in gws:
            gw["t_inj"] += t0
    elif "glitch_0" not in glitch_cfg and "gw_0" not in gw_cfg:
        t_injs = make_t_injs(
            glitch_rate=glitch_cfg["daily_rate"],
            gw_rate=gw_cfg["daily_rate"],
            duration=pipe_cfg["duration"].to("s").value,
            window=500,
        )

        glitches, gws = compute_anomalies(
            glitch_cfg=glitch_cfg,
            gw_cfg=gw_cfg,
            pipe_cfg=pipe_cfg,
            orbits_input_path=orbits_input_fn,
            t_injs=t_injs,
            dt=dt,
            size=size,
        )
    else:
        raise AttributeError("Mix-and-matching single and large-scale gw and glitch injections unsupported")

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
