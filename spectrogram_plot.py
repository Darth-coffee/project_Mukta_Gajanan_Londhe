import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, filtfilt


# Configuration

wav_dir =  '' # Fill in directory containing wav files
output_dir = '' # Fill in the output directory


#Signal Processing Utils


def bandpass_filter(data, fs, lowcut, highcut):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    return filtfilt(b, a, data)

def generate_spectrogram(y, fs, nfft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop_length, window='hann'))
    S_dB = 20 * np.log10(S + 1e-6)
    return S_dB

def save_spectrogram_image(S_dB, time_full, freq, output_filename):
    fig = plt.figure(figsize=(6, 2))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.imshow(S_dB, aspect='auto', origin='lower', cmap='jet', 
              extent=[time_full[0], time_full[-1], freq[0], freq[-1]])
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_windows_from_spectrogram(S_dB, window_width, window_height):
    windows = []
    n_frames = S_dB.shape[1]
    step_size = window_width
    
    for start_frame in range(0, n_frames - window_width, step_size):
        end_frame = start_frame + window_width
        spectrogram_window = S_dB[:, start_frame:end_frame]
        
        if spectrogram_window.shape[0] != window_height:
            spectrogram_window = np.resize(spectrogram_window, (window_height, window_width))
        
        windows.append(spectrogram_window)
        
    return windows


for filename in os.listdir(wav_dir):
    if filename.endswith('.wav'):
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(wav_dir, filename)
        
        y, fs = librosa.load(wav_path, sr=32000)
        y_filtered = bandpass_filter(y, fs, 300, 10000)
        
        S_dB = generate_spectrogram(y_filtered, fs)
        freq = librosa.fft_frequencies(sr=fs)
        time_full = librosa.frames_to_time(np.arange(S_dB.shape[1]), sr=fs, hop_length=512)
        
        windows = extract_windows_from_spectrogram(S_dB, window_width=600, window_height=200)
        
        for i, window in enumerate(windows):
            fig_filename = f"{base_name}_window_{i+1:02d}.png"
            
            # User needs to provide the path here
            if output_dir:  
                fig_path = os.path.join(output_dir, fig_filename)
                save_spectrogram_image(window, time_full, freq, fig_path)
                print(f"Saved spectrogram window: {fig_filename}")
            else:
                print(f"Please specify the output directory to save the images.")
