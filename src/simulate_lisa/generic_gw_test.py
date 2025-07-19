import os
import lisaglitch
from lisainstrument import Instrument
from pytdi.michelson import X2, Y2, Z2
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pytdi import Data
from scipy.signal.windows import tukey

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch_data/")
PATH_orbit_data = os.path.join(PATH_bethLISA, "dist/orbit_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_interferometer_plots = os.path.join(PATH_bethLISA,
                                         "dist/interferometer_plots/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")

t0 = 10368000
size = 86400 * 4
dt = 0.25
t_inj = t0 + 10000
t_rise = 5
t_fall = 20
level = 0

inj_points = [
    "readout_isi_carrier_12",
    "readout_isi_carrier_23",
    "readout_isi_carrier_31",
    "readout_isi_carrier_13",
    "readout_isi_carrier_32",
    "readout_isi_carrier_21",
]

for inj_point in inj_points:
    g = lisaglitch.OneSidedDoubleExpGlitch(
        inj_point=inj_point,
        t0=t0,
        size=size,
        dt=dt,
        t_inj=t_inj,
        t_rise=t_rise,
        t_fall=t_fall,
        level=level,
    )
    g.write(path=PATH_glitch_data + "gw_test.h5", mode="a")

output_txt = PATH_glitch_data + "gw_test.txt"

if os.path.exists(output_txt):
    os.remove(output_txt)

header = (
    "shape  "
    + "inj_point "
    + "size  "
    + "dt  "
    + "physics_upsampling  "
    + "t0  "
    + "t_inj "
    + "level "
    + "t_rise "
    + "t_fall "
)

with open(f"{output_txt}", "w") as f:
    f.write(header + "\n")

with open(f"{output_txt}", "a") as f:
    f.write(
        "GenericGWSignal" + "  "
        + "gw" + " "
        + str(size) + "  "
        + str(dt) + "  "
        + str(1) + "  "
        + str(t0) + "  "
        + str(t_inj) + " "
        + str(float(level)) + " "
        + str(t_rise) + " "
        + str(t_fall) + "\n"
    )

lisa_instrument = Instrument(
    size=size,
    dt=dt,
    t0=t0,
    orbits=PATH_orbit_data + "orbits.h5",
    physics_upsampling=1,
    aafilter=None,
    glitches=PATH_glitch_data + "gw_test.h5",
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
