import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pywt

# ── Load data ──────────────────────────────────────────────────────────────────
# CHANGE THIS PATH to your own CSV file
df = pd.read_csv('/Users/lenas/Documents/VIP Research/IMUTEST2_cleaned.csv')
time   = df['time'].values
signal = df['accel_z'].values * 9.81   # convert g's → m/s²
signal = signal - np.mean(signal)      # remove DC

dt_median = np.median(np.diff(time))
fs = 1.0 / dt_median

# ── Continuous Wavelet Transform (CWT) ────────────────────────────────────────
wavelet = 'cmor1.5-1.0'                        # complex Morlet wavelet
freqs_of_interest = np.linspace(0.5, 15, 200)  # frequency range to analyse (Hz)
scales = pywt.frequency2scale(wavelet, freqs_of_interest / fs)

coeffs, freqs_cwt = pywt.cwt(signal, scales, wavelet, sampling_period=dt_median)
power = np.abs(coeffs) ** 2                    # power = squared magnitude

# ── Plot ───────────────────────────────────────────────────────────────────────
# Can change the figure size, colors, and other parameters to your liking
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2.5]})
fig.patch.set_facecolor("#ffffff")

# Top: raw signal
ax1 = axes[0]
ax1.set_facecolor("#ffffff")
ax1.plot(time, signal, color="#4D9DFF", linewidth=0.7, alpha=0.9)
ax1.set_ylabel('Accel Z (m/s²)', color="#000000", fontsize=9)
ax1.set_title('Wavelet Analysis — Vertical Acceleration (accel_z)', color='white', fontsize=13, fontweight='bold', pad=10)
ax1.tick_params(colors="#000000", labelsize=8)
ax1.set_xlim(time[0], time[-1])
ax1.grid(True, color='#2a2d3a', linewidth=0.4)
for spine in ax1.spines.values():
    spine.set_edgecolor('#333344')

# Bottom: scalogram (time-frequency heatmap)
ax2 = axes[1]
ax2.set_facecolor("#ffffff")
#im = ax2.contourf(time, freqs_cwt, power, levels=100, cmap='inferno')
cmap_white = LinearSegmentedColormap.from_list('white_inferno', ['white', '#fcb045', '#ed1c5e', '#3a0ca3'])
im = ax2.contourf(time, freqs_cwt, power, levels=100, cmap=cmap_white)
ax2.set_ylabel('Frequency (Hz)', color="#0B0B0B", fontsize=9)
ax2.set_xlabel('Time (s)',        color="#000000", fontsize=9)
ax2.tick_params(colors="#000000", labelsize=8)
for spine in ax2.spines.values():
    spine.set_edgecolor("#FFFFFF")

cbar = fig.colorbar(im, ax=ax2, pad=0.01)
cbar.set_label('Power (m/s²)²', color="#000000", fontsize=8)
cbar.ax.tick_params(colors="#000000", labelsize=7)

plt.tight_layout()
# If you want to save the plot, uncomment the line below and specify your desired filename
#plt.savefig('imu_wavelet.png', dpi=150, bbox_inches='tight', facecolor="#ffffff")
#print("Plot saved → imu_wavelet.png")
plt.show()
