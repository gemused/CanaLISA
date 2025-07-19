import os
from lisagwresponse import ResponseFromStrain
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pytdi.michelson import X2, Y2, Z2
from pytdi import Data
import numpy as np
from lisainstrument import Instrument
from scipy.signal.windows import tukey

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_orbit_data = os.path.join(PATH_bethLISA, "dist/orbit_data/")
PATH_gw_data = os.path.join(PATH_bethLISA, "dist/gw_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_interferometer_plots = os.path.join(PATH_bethLISA,
                                         "dist/interferometer_plots/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")


class GWBurst(ResponseFromStrain):
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


t0 = 10368000
dt = 0.25
size = int(86400 / dt)
t_inj = t0 + 10000
t_rise = 5
t_fall = 20
level = 1e-10

orbits_path = PATH_orbit_data + "orbits.h5"
gw_output_path = PATH_gw_data + "gw.h5"
simulation_output_path = PATH_simulation_data + "gw_test.h5"

gw1 = GWBurst(
    t_rise=t_rise,
    t_fall=t_fall,
    level=level,
    t_inj=t_inj,
    orbits=orbits_path,
    gw_beta=0,
    gw_lambda=0,
    dt=dt,
    size=size,
    t0=t0,
)

gw2 = GWBurst(
    t_rise=t_rise,
    t_fall=t_fall,
    level=level,
    t_inj=t0,
    orbits=orbits_path,
    gw_beta=0,
    gw_lambda=0,
    dt=dt,
    size=size,
    t0=t0,
)

if os.path.exists(gw_output_path):
    os.remove(gw_output_path)

gw1.write(path=gw_output_path)
gw2.write(path=gw_output_path)

lisa_instrument = Instrument(
    size=size,
    dt=dt,
    t0=t0,
    orbits=orbits_path,
    physics_upsampling=1,
    aafilter=None,
    gws=gw_output_path,
)

lisa_instrument.disable_dopplers()

simulation_output_h5_path = PATH_simulation_data + "gw_test.h5"

if os.path.exists(simulation_output_h5_path):
    os.remove(simulation_output_h5_path)

lisa_instrument.write(simulation_output_h5_path)

lisa_instrument.plot_fluctuations(output=PATH_interferometer_plots
                                  + "fluctuations.png")

tdi_output_h5_path = PATH_tdi_data + "gw_test.h5"

channels = [X2, Y2, Z2]
tdi_names = ["X", "Y", "Z"]
tdi_dict = TimeSeriesDict()

data = Data.from_instrument(simulation_output_h5_path)
data.delay_derivative = None

for i in range(len(channels)):
    channel = channels[i]

    # CALCULATE TDI CHANNEL DATA
    tdi_data = channel.build(**data.args)(data.measurements)

    # WINDOW TDI CHANNEL DATA
    window = tukey(tdi_data.size, alpha=0.001)
    tdi_dict[tdi_names[i]] = TimeSeries(tdi_data * window, t0=t0, dt=dt)

# SAVE TDI CHANNEL DATA TO FILE
if os.path.exists(tdi_output_h5_path):
    os.remove(tdi_output_h5_path)

tdi_dict.write(tdi_output_h5_path, overwrite=True)
