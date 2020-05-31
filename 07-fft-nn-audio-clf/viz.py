
import librosa
from librosa import display
import matplotlib.pyplot as plt
import sys
import scipy
import numpy as np

# Take command linen arguments as filenames to be plotted
if len(sys.argv) > 1:
	filenames = sys.argv[1:]
else:
	filenames = ['./UrbanSound8K/audio/fold8/103076-3-0-0.wav']

# Visualize the FFT of a single audio file
def fftplot(audio, sr):
	N = len(audio)
	T = 1/sr
	y = scipy.fft.fft(audio)
	x = np.linspace(0., 1/(2*T), N//2)
	print(x.shape, y.shape)
	peaks = np.sort(np.abs(y[:N//2]))[:]
	print(peaks)
	plt.plot(x, 2./N * np.abs(y[:N//2]))
	plt.grid()
	plt.xlabel('Frequency')
	plt.ylabel('Magnitude')
	plt.show()

if __name__ == '__main__':
	for fn in filenames:
		samples, sampling_rate = librosa.load(fn)
		plt.figure()
		display.waveplot(y=samples, sr=sampling_rate)
		plt.xlabel('Time (s)')
		plt.ylabel('Amplitude')
		plt.show()
		fftplot(samples, sampling_rate)