
import librosa
from librosa import display
import matplotlib.pyplot as plt
import matplotlib
import sys
import scipy
import numpy as np

# Take command linen arguments as filenames to be plotted
if len(sys.argv) > 1:
    filenames = sys.argv[1:]
else:
    # Just choose a random sample for example if none are passed
    filenames = ['./UrbanSound8K/audio/fold8/103076-3-0-0.wav']


# Visualize the FFT of a single audio file
def fftplot(audio: np.ndarray,
            sr: int,
            ax: matplotlib.pyplot.Axes = None
            ) -> None:
    if ax is None:
        ax = plt
        fig, (ax0, ax1) = ax.subplots(nrows=2)
    else:
        ax0, ax1 = ax

    N = len(audio)
    T = 1/sr
    yf = scipy.fft.fft(audio)

    y = 2./N * np.abs(yf[:N//2])
    x = np.linspace(0., 1/(2*T), N//2)

    peaks, _ = scipy.signal.find_peaks(y, height=.0012)

    ax0.plot(y)
    ax0.plot(peaks, y[peaks], 'x')
    ax0.grid()
    ax0.set_xlabel('Frequency [Hz]')
    ax0.set_ylabel('Magnitude')
    ax0.set_title('FFT With Peaks')

    frequencies, times, spectrogram = scipy.signal.spectrogram(y, T)
    ax1.pcolormesh(times, frequencies, spectrogram)
    ax1.set_title('Spectrogram')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time [sec]')
    plt.tight_layout()

if __name__ == '__main__':
    for fn in filenames:
        samples, sampling_rate = librosa.load(fn)
        # Plot the raw data
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        
        display.waveplot(y=samples, sr=sampling_rate, ax=ax0)
        ax0.set_xlabel('Time [sec]')
        ax0.set_ylabel('Amplitude')
        ax0.set_title('Raw Audio Data')
        
        fftplot(samples, sampling_rate, ax=[ax1, ax2])
        plt.show()






















