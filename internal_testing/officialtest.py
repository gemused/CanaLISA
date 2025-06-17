import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from lisagwresponse import VerificationBinary
from lisainstrument import Instrument
from pytdi import Data
from pytdi.michelson import X2, Y2, Z2
# Use a standard set of Keplerian orbits
# These are provided by LISA Orbits
orbit_file = 'orbits.h5'
with h5py.File(orbit_file) as f:
    orbits_t0 = f.attrs['t0']
# Simulation runs for 3 days at 4 Hz,
# and starts 10 s after the orbit file
dt = 0.25 # s
fs = 1 / dt # Hz
duration = 60 * 60 * 24 * 3 # s
size = duration * fs # samples
t0 = orbits_t0 + 10 # s

print(1)

# Compute link responses to signal
source = VerificationBinary(
    period=569.4,
    distance=2089,
    masses=(0.8, 0.117),
    glong=57.7281,
    glat=6.4006,
    iota=60 * (np.pi / 180),
    orbits=orbit_file,
    size=size,
    dt=dt,
    t0=t0,
)

# Plot and write the link responses to disk
# source.plot(source.t[:8000])

print(2)

source.write('verification-binary.h5')

print("A")

# Define the instrumental setup, simulate
# and write the measurements to disk
instru = Instrument(
    orbits=orbit_file,
    # gws='verification-binary.h5',
    laser_shape='white',
    size=size,
    dt=dt,
    t0=t0,
)
instru.disable_all_noises(but='laser')

print(3)

instru.write('measurements.h5')

print(4)

# Compute TDI Michelson channels
data = Data.from_instrument('measurements.h5')
X2 = X2.build(**data.args)(data.measurements)
Y2 = Y2.build(**data.args)(data.measurements)
Z2 = Z2.build(**data.args)(data.measurements)

print(5)

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

# Plot the TDI Michelson channels
psd = lambda tseries: scipy.signal.welch(
    tseries,
    fs=fs,
    nperseg=2**16,
    window=('kaiser', 30),
    detrend=None
)
freq, X2_psd = psd(X2)
freq, Y2_psd = psd(Y2)
freq, Z2_psd = psd(Z2)
plt.loglog(freq, np.sqrt(X2_psd), label='X2')
plt.loglog(freq, np.sqrt(Y2_psd), label='Y2')
plt.loglog(freq, np.sqrt(Z2_psd), label='Z2')
plt.xlabel('Frequency [Hz]')
plt.ylabel('ASD [Hz/, Hz$^{-1/2}$]')
plt.legend()
plt.savefig("official_psd.png")

print(6)
