import os
import numpy as np
from scipy.signal.windows import tukey
from pytdi.michelson import X2, Y2, Z2
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from lisainstrument import Instrument
from pytdi import Data


PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_glitch_config = os.path.join(PATH_bethLISA, "dist/glitch/glitch_config/")
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch/glitch_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/lisa_data/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/lisa_data/tdi_data/")
PATH_orbits_data = os.path.join(PATH_bethLISA, "dist/lisa_data/orbits_data/")
PATH_interferometer_plots = os.path.join(PATH_bethLISA,
                                         "dist/lisa_data/interferometer_plots/")
PATH_pipe_data = os.path.join(PATH_bethLISA, "dist/pipe/pipe_data/")
PATH_gw_data = os.path.join(PATH_bethLISA, "dist/gw/gw_data/")


def init_pipe(pipe_input_path):
    pipe_input = np.genfromtxt(pipe_input_path, dtype=float)

    return pipe_input[0], pipe_input[1], pipe_input[2]


def simulate_lisa(
    glitch_input_path, gw_input_path, orbits_input_path,
    simulation_output_path, t0, dt, size, disable_noise
):
    # CREATE LISA INSTRUMENT OBJECT
    lisa_instrument = Instrument(
        size=size,
        dt=dt,
        t0=t0,
        orbits=orbits_input_path,
        aafilter=None,
        glitches=glitch_input_path,
        gws=gw_input_path,
    )

    lisa_instrument.disable_dopplers()

    if disable_noise:
        lisa_instrument.disable_all_noises()

    # SIMULATE LISA AND SAVE RESULTS TO FILE
    if os.path.exists(simulation_output_path):
        os.remove(simulation_output_path)

    lisa_instrument.write(simulation_output_path)

    lisa_instrument.plot_fluctuations(output=PATH_interferometer_plots
                                      + "fluctuations.png")


def compute_tdi(
    simulation_input_path, tdi_output_path, t0, dt
):
    channels = [X2, Y2, Z2]
    tdi_names = ["X", "Y", "Z"]
    tdi_dict = TimeSeriesDict()

    # GET DATA FROM LISA INSTRUMENT
    data = Data.from_instrument(simulation_input_path)
    data.delay_derivative = None

    for i in range(len(channels)):
        channel = channels[i]

        # CALCULATE TDI CHANNEL DATA
        tdi_data = channel.build(**data.args)(data.measurements)

        # WINDOW TDI CHANNEL DATA
        window = tukey(tdi_data.size, alpha=0.001)
        tdi_dict[tdi_names[i]] = TimeSeries(tdi_data * window, t0=t0, dt=dt)

    # SAVE TDI CHANNEL DATA TO FILE
    if os.path.exists(tdi_output_path):
        os.remove(tdi_output_path)

    tdi_dict.write(tdi_output_path, overwrite=True)


def inject_anomalies(
    glitch_input_fn, gw_input_fn, pipe_input_fn, orbits_input_fn,
    simulation_output_fn, tdi_output_fn, disable_noise=False
):
    t0, dt, size = init_pipe(
        pipe_input_path=PATH_pipe_data + pipe_input_fn + ".txt",
    )

    simulate_lisa(
        glitch_input_path=PATH_glitch_data + glitch_input_fn + ".h5",
        gw_input_path=PATH_gw_data + gw_input_fn + ".h5",
        orbits_input_path=PATH_orbits_data + orbits_input_fn + ".h5",
        simulation_output_path=PATH_simulation_data + simulation_output_fn + ".h5",
        t0=t0,
        dt=dt,
        size=size,
        disable_noise=disable_noise,
    )

    compute_tdi(
        simulation_input_path=PATH_simulation_data + simulation_output_fn + ".h5",
        tdi_output_path=PATH_tdi_data + tdi_output_fn + ".h5",
        t0=t0,
        dt=dt,
    )
