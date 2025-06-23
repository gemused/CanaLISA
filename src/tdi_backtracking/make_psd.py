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

f_isi_12, isi_12_psd = welch(isi_12_data, fs=4, nperseg=345600)
f_tmi_12, tmi_12_psd = welch(tmi_12_data, fs=4, nperseg=345600)
f_rfi_12, rfi_12_psd = welch(rfi_12_data, fs=4, nperseg=345600)

fs_isi_12 = f_isi_12[1] - f_isi_12[0]
fs_tmi_12 = f_tmi_12[1] - f_tmi_12[0]
fs_rfi_12 = f_rfi_12[1] - f_rfi_12[0]

print(fs_tmi_12)

isi_12_psd_fs = FrequencySeries(isi_12_psd, delta_f=fs_isi_12)
tmi_12_psd_fs = FrequencySeries(tmi_12_psd, delta_f=fs_tmi_12)
rfi_12_psd_fs = FrequencySeries(rfi_12_psd, delta_f=fs_rfi_12)

if os.path.exists(PATH_psd_data + "isi_psd.hdf"):
    os.remove(PATH_psd_data + "isi_psd.hdf")

if os.path.exists(PATH_psd_data + "tmi_psd.hdf"):
    os.remove(PATH_psd_data + "tmi_psd.hdf")

if os.path.exists(PATH_psd_data + "rfi_psd.hdf"):
    os.remove(PATH_psd_data + "rfi_psd.hdf")

isi_12_psd_fs.save(PATH_psd_data + "isi_psd.hdf")
tmi_12_psd_fs.save(PATH_psd_data + "tmi_psd.hdf")
rfi_12_psd_fs.save(PATH_psd_data + "rfi_psd.hdf")
