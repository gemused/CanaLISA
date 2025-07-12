from gwpy.timeseries import TimeSeriesDict
from gwpy.plot import Plot
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import os

#get tdi output file
print(os.getcwd())
tdi_path = os.path.join(os.getcwd(), "dist/tdi_data/0.h5")
XYZ_fig_path = os.path.join(os.getcwd(), "dist/interferometer_plots/XYZ_plot")
AET_fig_path = os.path.join(os.getcwd(), "dist/interferometer_plots/AET_plot")
OVERLAY_fig_path = os.path.join(os.getcwd(), "dist/interferometer_plots/OVERLAY_plot")
PSD_fig_path = os.path.join(os.getcwd(), "dist/interferometer_plots/PSD_plot")
W_PSD_fig_path = os.path.join(os.getcwd(), "dist/interferometer_plots/W_PSD_plot")


tdiseries = TimeSeriesDict.read(tdi_path)

#initialise plot 
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)

#X plot
ax1.plot(tdiseries["X"])
ax1.set_ylabel("X")
ax1.grid(False)
ax1.tick_params('x', which='both', bottom=False, top=True, direction='in', labelsize=10)
ax1.tick_params('y', left=True, right=True, direction='in', labelsize=10)

#Y plot
ax2.plot(tdiseries["Y"])
ax2.set_ylabel("Y")
ax2.grid(False)
ax2.tick_params('x', which='both', bottom=False, top=False, direction='in', labelsize=10)
ax2.tick_params('y', left=True, right=True, direction='in', labelsize=10)

#Z plot
ax3.plot(tdiseries["Z"])
ax3.set_ylabel("Z")
ax3.set_xlabel("Time (seconds)")
ax3.grid(False)
ax3.tick_params('x', which='both', bottom=True, top=False, direction='in', labelsize=10)
ax3.tick_params('y', left=True, right=True, direction='in', labelsize=10)
ax3.xaxis.set_major_locator(MultipleLocator(len(tdiseries["X"])/20))

plt.grid(False)
fig.tight_layout()
fig.savefig(XYZ_fig_path)

#initialise plot 
fig2, (ax4, ax5, ax6) = plt.subplots(3,1, sharex=True)

#A plot
ax4.plot((1/np.sqrt(2))*(tdiseries["Z"]-tdiseries["X"]))
ax4.set_ylabel("A")
ax4.grid(False)
ax4.tick_params('x', which='both', bottom=False, top=True, direction='in', labelsize=10)
ax4.tick_params('y', left=True, right=True, direction='in', labelsize=10)

#E plot
ax5.plot((1/np.sqrt(6))*(tdiseries["X"]-2*tdiseries["Y"]+tdiseries["Z"]))
ax5.set_ylabel("E")
ax5.grid(False)
ax5.tick_params('x', which='both', bottom=False, top=False, direction='in', labelsize=10)
ax5.tick_params('y', left=True, right=True, direction='in', labelsize=10)

#T plot
ax6.plot((1/np.sqrt(3))*(tdiseries["X"] + tdiseries["Y"] + tdiseries["Z"]))
ax6.set_ylabel("T")
ax6.set_xlabel("Time (seconds)")
ax6.grid(False)
ax6.tick_params('x', which='both', bottom=True, top=False, direction='in', labelsize=10)
ax6.tick_params('y', left=True, right=True, direction='in', labelsize=10)
ax6.xaxis.set_major_locator(MultipleLocator(len(tdiseries["X"])/20))

plt.grid(False)
fig2.tight_layout()
fig2.savefig(AET_fig_path)



#alternately use gwpy.plot.Plot to plot the timeseries more easily
fig3 = plt.figure()
plt.grid(False)
fig3 = Plot(tdiseries["X"].whiten(1000,400), tdiseries["Y"].whiten(1000,400), tdiseries["Z"].whiten(1000,400), separate=True, sharex=True)
ax = fig3.gca()
ax.set_xlim(-3600,90000)
fig3.tight_layout()
fig3.savefig(OVERLAY_fig_path)




#plot PSD of X channel and whitened PSD too
fig4, (ax7, ax8) = plt.subplots(2,1, figsize=(8,8))

tdipsd = tdiseries["Z"].psd(1000,400)
w_tdipsd = tdiseries["Z"].whiten(1000,400).psd(3000,1200)

ax7.loglog(tdipsd)
ax7.set_xlim(0.001, 1)
ax7.grid(False)
ax7.set_ylabel('Power')

ax8.loglog(w_tdipsd)
ax8.set_xlim(0.001, 1)
ax8.grid(False)
ax8.set_ylabel('Whitened Power')
ax8.set_xlabel('Frequency (Hz)')

fig4.tight_layout()
fig4.savefig(PSD_fig_path)


