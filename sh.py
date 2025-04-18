import numpy as np
import matplotlib.pyplot as plt

# ✨ Your EEG signal (178 values)
eeg_signal = [
801,-56,-20,21,71,72,48,32,23,0,-26,-23,-12,-5,16,30,17,20,36,26,-15,-52,-68,-67,-33,18,56,68,55,50,34,26,48,61,50,18,0,-11,-5,12,25,31,45,60,55,47,30,-18,-92,-125,-127,-112,-59,-9,6,0,-5,-17,-46,-71,-78,-83,-59,10,65,88,90,93,69,32,14,-3,-9,6,33,58,64,83,79,61,58,57,33,6,-1,-23,-52,-58,-34,-14,-2,23,26,-5,-41,-69,-105,-122,-120,-115,-95,-64,-25,-1,21,52,60,52,52,49,6,-25,-43,-47,-35,-14,14,25,39,37,10,-34,-49,-56,-71,-60,-29,-10,11,54,80,75,61,51,47,53,72,72,47,3,-25,-27,-26,-13,0,2,22,39,21,-3,-12,-17,-29,-25,-2,5,17,54,61,40,31,56,69,66,65,43,30,36,66,80,91,98,61,-10,-52,-97,-153,-162
]

# Convert to NumPy array
eeg_signal = np.array(eeg_signal)

# Plot Time Domain
plt.figure(figsize=(12, 5))
plt.plot(eeg_signal, label='EEG Signal (Time Domain)', color='teal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('EEG Signal in Time Domain')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- FFT Part ---

# Compute FFT
fft_values = np.fft.fft(eeg_signal)
fft_freqs = np.fft.fftfreq(len(eeg_signal), d=1)  # Assuming 1 sample/sec

# Only take the positive frequencies
positive_freqs = fft_freqs[:len(fft_freqs)//2]
positive_magnitude = np.abs(fft_values[:len(fft_values)//2])

# Plot Frequency Domain
plt.figure(figsize=(12, 5))
plt.plot(positive_freqs, positive_magnitude, color='purple')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of EEG Signal (FFT)')
plt.grid(True)
plt.tight_layout()
plt.show()

