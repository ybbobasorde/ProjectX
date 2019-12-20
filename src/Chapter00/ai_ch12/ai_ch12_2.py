#
# Speech recognition: Audio Generator
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from io import BytesIO
from base64 import b64encode

# Output file where the audio will be saved 
output_file = 'data/ai_ch12/generated_audio.wav'

# Specify audio parameters
duration = 4  # in seconds
sampling_freq = 44100  # in Hz
tone_freq = 784 
min_val = -4 * np.pi
max_val = 4 * np.pi

# Generate the audio signal
t = np.linspace(min_val, max_val, duration * sampling_freq)
signal = np.sin(2 * np.pi * tone_freq * t)

# Add some noise to the signal
noise = 0.5 * np.random.rand(duration * sampling_freq)
signal += noise

# Scale it to 16-bit integer values
scaling_factor = np.power(2, 15) - 1
signal_normalized = signal / np.max(np.abs(signal))
signal_scaled = np.int16(signal_normalized * scaling_factor)

# Save the audio signal in the output file 
write(output_file, sampling_freq, signal_scaled)

# Extract the first 200 values from the audio signal 
signal = signal[:200]

# Construct the time axis in milliseconds
time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq) 

# Plot the audio signal
plt.plot(time_axis, signal, color='black')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Generated audio signal')

#plt.show()
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai12_plot_url1 = b64encode(img.getvalue()).decode('ascii')
