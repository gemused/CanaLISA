import os
import h5py
from scipy.signal import welch
from pycbc.types.frequencyseries import FrequencySeries

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_psd_data = os.path.join(PATH_bethLISA, "dist/psd_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")

with h5py.File(PATH_simulation_data + "simulation_output_no_glitch.h5", "r") as sim_file:
    isi_12_data = sim_file["isi_carrier_fluctuations"]["12"]
    tmi_12_data = sim_file["tmi_carrier_fluctuations"]["12"]
    rfi_12_data = sim_file["rfi_carrier_fluctuations"]["12"]
with h5py.File(PATH_tdi_data + "tdi_output_no_glitch.h5", "r") as tdi_file:
    X_data = tdi_file["X"][:]
    Y_data = tdi_file["Y"][:]
    Z_data = tdi_file["Z"][:]
    XY_data = [(X_data[i] + Y_data[i]) / 2 for i in range(len(X_data))]
    YZ_data = [(Y_data[i] + Z_data[i]) / 2 for i in range(len(Y_data))]
    XZ_data = [(X_data[i] + Z_data[i]) / 2 for i in range(len(X_data))]

f_isi_12, isi_12_psd = welch(isi_12_data, fs=4, nperseg=345600)
f_tmi_12, tmi_12_psd = welch(tmi_12_data, fs=4, nperseg=345600)
f_rfi_12, rfi_12_psd = welch(rfi_12_data, fs=4, nperseg=345600)
f_X, X_psd = welch(X_data, fs=4, nperseg=345600)
f_Y, Y_psd = welch(Y_data, fs=4, nperseg=345600)
f_Z, Z_psd = welch(Z_data, fs=4, nperseg=345600)
f_XY, XY_psd = welch(XY_data, fs=4, nperseg=345600)
f_YZ, YZ_psd = welch(YZ_data, fs=4, nperseg=345600)
f_XZ, XZ_psd = welch(XZ_data, fs=4, nperseg=345600)

fs_isi_12 = f_isi_12[1] - f_isi_12[0]
fs_tmi_12 = f_tmi_12[1] - f_tmi_12[0]
fs_rfi_12 = f_rfi_12[1] - f_rfi_12[0]
fs_X = f_X[1] - f_X[0]
fs_Y = f_Y[1] - f_Y[0]
fs_Z = f_Z[1] - f_Z[0]
fs_XY = f_XY[1] - f_XY[0]
fs_YZ = f_YZ[1] - f_YZ[0]
fs_XZ = f_XZ[1] - f_XZ[0]

isi_12_psd_fs = FrequencySeries(isi_12_psd, delta_f=fs_isi_12)
tmi_12_psd_fs = FrequencySeries(tmi_12_psd, delta_f=fs_tmi_12)
rfi_12_psd_fs = FrequencySeries(rfi_12_psd, delta_f=fs_rfi_12)
X_psd_fs = FrequencySeries(X_psd, delta_f=fs_X)
Y_psd_fs = FrequencySeries(Y_psd, delta_f=fs_Y)
Z_psd_fs = FrequencySeries(Z_psd, delta_f=fs_Z)
XY_psd_fs = FrequencySeries(XY_psd, delta_f=fs_XY)
YZ_psd_fs = FrequencySeries(YZ_psd, delta_f=fs_YZ)
XZ_psd_fs = FrequencySeries(XZ_psd, delta_f=fs_XZ)

if os.path.exists(PATH_psd_data + "isi_psd.hdf"):
    os.remove(PATH_psd_data + "isi_psd.hdf")
if os.path.exists(PATH_psd_data + "tmi_psd.hdf"):
    os.remove(PATH_psd_data + "tmi_psd.hdf")
if os.path.exists(PATH_psd_data + "rfi_psd.hdf"):
    os.remove(PATH_psd_data + "rfi_psd.hdf")
if os.path.exists(PATH_psd_data + "X_psd.hdf"):
    os.remove(PATH_psd_data + "X_psd.hdf")
if os.path.exists(PATH_psd_data + "Y_psd.hdf"):
    os.remove(PATH_psd_data + "Y_psd.hdf")
if os.path.exists(PATH_psd_data + "Z_psd.hdf"):
    os.remove(PATH_psd_data + "Z_psd.hdf")
if os.path.exists(PATH_psd_data + "XY_psd.hdf"):
    os.remove(PATH_psd_data + "XY_psd.hdf")
if os.path.exists(PATH_psd_data + "YZ_psd.hdf"):
    os.remove(PATH_psd_data + "YZ_psd.hdf")
if os.path.exists(PATH_psd_data + "XZ_psd.hdf"):
    os.remove(PATH_psd_data + "XZ_psd.hdf")

isi_12_psd_fs.save(PATH_psd_data + "isi_psd.hdf")
tmi_12_psd_fs.save(PATH_psd_data + "tmi_psd.hdf")
rfi_12_psd_fs.save(PATH_psd_data + "rfi_psd.hdf")
X_psd_fs.save(PATH_psd_data + "X_psd.hdf")
Y_psd_fs.save(PATH_psd_data + "Y_psd.hdf")
Z_psd_fs.save(PATH_psd_data + "Z_psd.hdf")
XY_psd_fs.save(PATH_psd_data + "XY_psd.hdf")
YZ_psd_fs.save(PATH_psd_data + "YZ_psd.hdf")
XZ_psd_fs.save(PATH_psd_data + "XZ_psd.hdf")
