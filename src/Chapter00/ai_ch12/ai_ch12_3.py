#
# Speech recognition: Audio Plotter
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from io import BytesIO
from base64 import b64encode

# Read the audio file
sampling_freq, signal = wavfile.read('data/ai_ch12/random_sound.wav')

# Display the params
print('\nSignal shape:', signal.shape)
ai12_3_url1 = signal.shape
print('Datatype:', signal.dtype)
ai12_3_url2 = signal.dtype
print('Signal duration:', round(signal.shape[0] / float(sampling_freq), 2), 'seconds')
ai12_3_url3_str = round(signal.shape[0] / float(sampling_freq), 2), 'seconds'
ai12_3_url3 = ai12_3_url3_str

# Normalize the signal 
signal = signal / np.power(2, 15)

# Extract the first 50 values
signal = signal[:50]

# Construct the time axis in milliseconds
time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq)

# Plot the audio signal
plt.plot(time_axis, signal, color='black')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')

#plt.show()
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai12_plot_url1 = b64encode(img.getvalue()).decode('ascii')