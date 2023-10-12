import csv
import mne
import glob
import json
import pywt
import random
import os, sys
import warnings
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from scipy import stats
from scipy.signal import stft
from collections import Counter
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import periodogram, welch
from mne.time_frequency import psd_array_multitaper
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


mne.set_log_level('ERROR')
import logging
from mne.utils import set_log_level
logging.basicConfig(level=logging.ERROR)
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def plot_channels_comparison(raw_list, start_time, end_time, channels_to_display=None, names=None, save=None):
    """
    Plot the comparison of channels across multiple raw EEG data files.

    Parameters:
        raw_list (list): List of mne.io.eeglab.eeglab.RawEEGLAB objects representing the raw EEG data.
        start_time (float): Start time (in seconds) for the visualization.
        end_time (float): End time (in seconds) for the visualization.
        channels_to_display (list, optional): List of channel names to display. If not provided, all channels are displayed.
        names (list, optional): List of names for each raw file. If not provided, default names are used (Raw 1, Raw 2, ...).
        save (str, optional): Name of the PNG file to save the plot. If provided, the plot is saved with the given name (without extension).

    Returns:
        None

    Raises:
        None

    Example:
        raw_list = [
            mne.io.read_raw_eeglab('path/to/your/file1.set'),
            mne.io.read_raw_eeglab('path/to/your/file2.set'),
            mne.io.read_raw_eeglab('path/to/your/file3.set'),
        ]
        start_time = 5.0  # Start time in seconds
        end_time = 10.0  # End time in seconds

        # Example usage with optional parameters
        plot_channels_comparison(raw_list, start_time, end_time, names=['Subject 1', 'Subject 2', 'Subject 3'], save='output_plot')
    """
    def get_start_end(raw):
        # Get the sampling frequency for the first raw object
        sfreq = raw.info['sfreq']

        # Convert start and end time to samples
        start_sample = int(start_time * sfreq)
        end_sample = int(end_time * sfreq)
        return start_sample, end_sample

    # Determine the channels to display
    if channels_to_display is None:
        channels_to_display = raw_list[0].ch_names

    # Determine the names for each raw object
    if names is None:
        names = [f"Raw {i+1}" for i in range(len(raw_list))]

    # Create subplots
    num_channels = len(channels_to_display)
    num_files = len(raw_list)
    fig, axes = plt.subplots(num_channels, num_files, figsize=(5*num_files, 2*num_channels), sharex=True, sharey=True)

    # Single file and single channel case
    if num_files == 1 and num_channels == 1:
        raw = raw_list[0]
        start_sample, end_sample = get_start_end(raw)
        channel = channels_to_display[0]
        if channel in raw.ch_names:
            channel_data = raw.copy().pick_channels([channel])
            channel_signal = channel_data.get_data()[0]
            channel_signal_range = channel_signal[start_sample:end_sample]
            time_range = channel_data.times[start_sample:end_sample]
            axes.plot(time_range, channel_signal_range)
            axes.set_title(f"{channel} ({names[0]})")
            axes.set_ylabel('Amplitude')

    # Single file and multi-channel case
    elif num_files == 1 and num_channels > 1:
        raw = raw_list[0]
        start_sample, end_sample = get_start_end(raw)
        for i, channel in enumerate(channels_to_display):
            if channel in raw.ch_names:
                channel_data = raw.copy().pick_channels([channel])
                channel_signal = channel_data.get_data()[0]
                channel_signal_range = channel_signal[start_sample:end_sample]
                time_range = channel_data.times[start_sample:end_sample]
                axes[i].plot(time_range, channel_signal_range)
                axes[i].set_title(f"{channel} ({names[0]})")
                axes[i].set_ylabel('Amplitude')

    # Multi-file and multi-channel case
    else:
        for j, raw in enumerate(raw_list):
            for i, channel in enumerate(channels_to_display):
                start_sample, end_sample = get_start_end(raw)
                if channel in raw.ch_names:
                    channel_data = raw.copy().pick_channels([channel])
                    channel_signal = channel_data.get_data()[0]
                    channel_signal_range = channel_signal[start_sample:end_sample]
                    time_range = channel_data.times[start_sample:end_sample]
                    if len(channels_to_display) > 1:
                        axes[i, j].plot(time_range, channel_signal_range)
                        axes[i, j].set_title(f"{channel} ({names[j]})")
                        axes[i, j].set_ylabel('Amplitude')
                    else:
                        axes[j].plot(time_range, channel_signal_range)
                        axes[j].set_title(f"{channel} ({names[j]})")
                        axes[j].set_ylabel('Amplitude')

    # Set the x-axis label
    if num_files > 1:
        if len(channels_to_display) > 1:
            for j in range(num_files):
                axes[-1, j].set_xlabel('Time (s)')
        else:
            axes[-1].set_xlabel('Time (s)')
    else:
        if len(channels_to_display) > 1: 
            axes[-1].set_xlabel('Time (s)')
        else:
            axes.set_xlabel('Time (s)')



    # Adjust the layout
    plt.tight_layout()

    # Save the high-quality PNG file if save is provided
    if save is not None:
        plt.savefig(save + '.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()




class DS004504(Dataset):
    def __init__(self, root, preload=None):
        super().__init__()
        if preload == None:
            subjects = DS004504.load_participants(root+"participants.tsv")
            subjects = self.load_meta_data(root, subjects)
            
            self.subjects = subjects
        else:
            self.subjects = preload

    @staticmethod
    def load_participants(filename):
        data = []
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader: 
                data.append(row)
        return data

    def load_meta_data(self, root, subjects):
        for participant in subjects:
            json_file = glob.glob(root + participant['participant_id'] + '/eeg/*.json')[0]
            tsv_file = glob.glob(root + participant['participant_id'] + '/eeg/*.tsv')[0]
            set_file = glob.glob(root + participant['participant_id'] + '/eeg/*.set')[0]
            with open(json_file) as f_json: 
                json_data = json.load(f_json)
                participant['json'] = json_data
            with open(tsv_file, newline='') as t_csv:
                tsv_data = csv.DictReader(t_csv, delimiter='\t')
                participant['channels'] = []
                for row in tsv_data:
                    participant['channels'].append(row)
            participant['signal_path'] = set_file
            participant['raw_path'] =set_file 
            participant['filtered_path'] = set_file.replace('ds004504/', 'ds004504/derivatives/')
            participant['preprocessed_path'] = set_file.replace('ds004504/', 'ds004504/preprocessed/')
                                                        
        return subjects
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, index, preload=None):
        
        if preload is None: 
            return self.subjects[index]
        
        
        subject = self.subjects[index]
        raw = mne.io.read_raw_eeglab(subject['filtered_path'], preload=False)
        subject['raw_data'] = raw
        return subject
    
    
class Subject():
    def __init__(self, participantId, gender, age, group, mmse, raw_path, filtered_path ):
        """
        Initializes a Subject instance representing an individual participant in an EEG study.
        
        Parameters:
        - participantId: Unique identifier for the participant.
        - gender: Gender of the participant.
        - age: Age of the participant.
        - group: Group to which the participant belongs.
        - mmse: Mini-Mental State Examination score of the participant.
        - raw_path: Path to the raw EEG recording file for the participant.
        - filtered_path: Path to the filtered EEG recording file for the participant.
        """
        self.participantId = participantId
        self.gender = gender
        self.age = age
        self.group = group
        self.mmse = mmse
        self.raw_path = raw_path
        self.filtered_path = filtered_path

    def get_participantId(self):
        """
        Returns the participant's unique identifier.
        """
        return self.participantId

    def get_gender(self):
        """
        Returns the participant's gender.
        """
        return self.gender
    
    def get_age(self):
        """
        Returns the participant's age.
        """
        return self.age

    def get_group(self):
        """
        Returns the group to which the participant belongs.
        """
        return self.group

    def get_mmse(self):
        """
        Returns the participant's Mini-Mental State Examination (MMSE) score.
        """
        return self.mmse

    def get_raw_path(self):
        """
        Returns the path to the raw EEG recording file for the participant.
        """
        return self.raw_path

    def get_filtered_path(self):
        """
        Returns the path to the filtered EEG recording file for the participant.
        """
        return self.filtered_path


class Epoch():
    def __init__(self, subject, recording_path, recording=None): 
        """
        Initializes an Epoch instance.
        
        Parameters:
        - subject: The subject object from which the epoch is extracted.
        - recording_path: Path to the raw EEG recording file.
        - start: The start time (in seconds) of the epoch.
        - end: The end time (in seconds) of the epoch.
        """
        self.recording_path = recording_path

        self.subject = subject
        self.label = self.subject.group
        self.recording = recording
        self.channel_names = self.recording.info['ch_names']

    def __getitem__(self, index):
        if isinstance(index, int) and 0 <= index < len(self.recording.info['ch_names']):
            return self.recording.get_data(picks=[index]).squeeze()
        
        if isinstance(index, str) and index in self.recording.info['ch_names']:
            return self.recording.get_data(picks=[index]).squeeze()

        return None
    
    def plot_cwt(self, index):
        signal = self.__getitem__(index)
                           
        # Define the wavelet and scales
        wavelet = 'morl'  # Choose a wavelet (e.g., Morlet)
        scales = np.arange(1, 128)  # Adjust the range of scales as needed

        # Perform Continuous Wavelet Transform (CWT)
        coeffs, freqs = pywt.cwt(signal, scales, wavelet)

        # Create a figure with two vertically aligned subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot the scaleogram in the top subplot
        im = ax1.imshow(np.abs(coeffs), extent=[0, len(signal), min(scales), max(scales)],
                aspect='auto', cmap='jet', interpolation='bilinear')
        ax1.set_title(f'Scaleogram of {self.label}')
        ax1.set_ylabel('Scale')
        ax1.grid(True)

        # Add a colorbar to the scaleogram subplot
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Magnitude')

        # Plot the signal in the bottom subplot
        ax2.plot(signal)
        ax2.set_title(f'{self.label} Signal')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

        plt.tight_layout()  # Ensures proper spacing between subplots
        plt.show()
                       
    def getEpochRecording(self):
        """
        Loads and crops the raw EEG recording from the given recording_path between 
        the specified start and end times.
        
        Returns:
        - raw_crop: The cropped raw EEG data.
        """


        return self.recording
    
    
    def plot_channel(self, index):
        channel_name = self.recording.info['ch_names'][index]
        
        # Get the data using the get_data method and select the specific channel using the index
        data = self.recording.get_data(picks=[index])
        
        # Get the times array
        times = self.recording.times
        
        # Plot the data using Matplotlib
        plt.plot(times, data[0, 0, :])
        plt.xlabel('Time (s)')
        plt.ylabel('MEG Data (T)')
        plt.title(f'Channel: {channel_name}')
        plt.grid(True)
        plt.show()
        
    def plot_all_channels(self):
        # Get the data and the number of channels
        data = self.recording.get_data()
        n_channels = len(self.recording.info['ch_names'])
        
        # Create a new figure
        plt.figure(figsize=(10, 2 * n_channels))
        
        # Iterate over all channels and plot them
        for i in range(n_channels):
            channel_name = self.recording.info['ch_names'][i]
            
            # Create a new subplot for each channel
            plt.subplot(n_channels, 1, i + 1)
            
            # Plot the data for the current channel
            plt.plot(self.recording.times, data[0, i, :])
            plt.xlabel('Time (s)')
            plt.ylabel('MEG Data (T)')
            plt.title(f'Channel: {channel_name}')
            plt.grid(True)
        
        # Adjust the layout so that plots do not overlap
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
        
    def plot_shortFtransform(self, index):
        recording = self.recording
        fs = recording.info['sfreq']
        channel = recording.get_data(picks=[index])
        frequencies, times, Zxx = stft(channel, fs=fs, nperseg=256, noverlap=128)
        
        Zxx = Zxx.squeeze()
        plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Spectrogram {recording.info["ch_names"][index]   }  ')
        plt.colorbar(label='Intensity [dB]')
        plt.show()
        
        
    def plot_all_shortFtransforms(self):
        recording = self.recording
        n_channels = len(recording.info['ch_names'])
        n_cols = 4
        n_rows = int(np.ceil(n_channels / n_cols))
        
        fs = recording.info['sfreq']
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        
        for i in range(n_channels):
            ax = axes[i // n_cols, i % n_cols]
            
            channel = recording.get_data(picks=[i])
            frequencies, times, Zxx = stft(channel.squeeze(), fs=fs, nperseg=256, noverlap=128)
            
            # Limit the maximum frequency to 50 Hz
            freq_idx = frequencies <= 50
            frequencies = frequencies[freq_idx]
            Zxx = Zxx[freq_idx, :]
            
            c = ax.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [sec]')
            ax.set_title(recording.info["ch_names"][i])
            
            fig.colorbar(c, ax=ax, label='Intensity [dB]')
        
        # Hide empty subplots
        for i in range(n_channels, n_rows * n_cols):
            fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.show()
        
    def plot_FFT(self, index):
        recording = self.recording
        channel = recording.get_data(picks=[0])
        channel = channel.copy()
        channel = channel.squeeze()
        # Get the times array
        times = recording.times

        # Step 4: Apply FFT
        frequencies = np.fft.fftfreq(len(times), d=times[1] - times[0])
        fft_res = np.fft.fft(channel)

        # Step 5: Create Spectrogram
        plt.specgram(channel, NFFT=256, Fs=recording.info['sfreq'], noverlap=128, cmap='inferno')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Spectrogram of Channel {recording.info["ch_names"][index]} ')
        plt.colorbar(label='Intensity [dB]')
        plt.show()



    def plot_all_FFT(self):
        recording = self.recording
        n_channels = len(recording.info['ch_names'])
        n_cols = 5
        n_rows = int(np.ceil(n_channels / n_cols))
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(21, 3 * n_rows))
        
        for index in range(n_channels):
            row = index // n_cols
            col = index % n_cols
            
            channel = recording.get_data(picks=[index])
            channel = channel.copy()
            channel = channel.squeeze()
            
            # Get the times array
            times = recording.times

            # Apply FFT
            frequencies = np.fft.fftfreq(len(times), d=times[1] - times[0])
            fft_res = np.fft.fft(channel)

            # Create Spectrogram
            cax = axes[row, col].specgram(channel, NFFT=256, Fs=recording.info['sfreq'], noverlap=128, cmap='inferno')
            axes[row, col].set_ylabel('Frequency [Hz]')
            axes[row, col].set_xlabel('Time [sec]')
            axes[row, col].set_title(f'Ch {recording.info["ch_names"][index]}')
            
            fig.colorbar(cax[3], ax=axes[row, col], label='Intensity [dB]')
        
        # Remove empty subplots
        for index in range(n_channels, n_rows * n_cols):
            fig.delaxes(axes.flatten()[index])
        
        plt.tight_layout()
        plt.show()
    
    def channel_mean(self, index):
        data = self.recording.get_data(picks=[index])
        mean_value = np.mean(data)
        return mean_value

    def channel_median(self, index):
        data = self.recording.get_data(picks=[index])
        median_value = np.median(data)
        return median_value

    def channel_mode(self, index):
        data = self.recording.get_data(picks=[index])
        mode_value = stats.mode(data, axis=None).mode[0]
        return mode_value

    def channel_variance(self, index):
        data = self.recording.get_data(picks=[index])
        variance_value = np.var(data)
        return variance_value

    def channel_std_dev(self, index):
        data = self.recording.get_data(picks=[index])
        std_dev_value = np.std(data)
        return std_dev_value

    def channel_skewness(self, index):
        data = self.recording.get_data(picks=[index])
        skewness_value = stats.skew(data, axis=None)
        return skewness_value

    def channel_kurtosis(self, index):
        data = self.recording.get_data(picks=[index])
        kurtosis_value = stats.kurtosis(data, axis=None)
        return kurtosis_value

    def channel_iqr(self, index):
        data = self.recording.get_data(picks=[index])
        iqr_value = stats.iqr(data, axis=None)
        return iqr_value

    def channel_entropy(self, index):
        data = self.recording.get_data(picks=[index])
        entropy_value = stats.entropy(data, axis=None)
        return entropy_value

    def channel_min(self, index):
        data = self.recording.get_data(picks=[index])
        min_value = np.min(data)
        return min_value

    def channel_max(self, index):
        data = self.recording.get_data(picks=[index])
        max_value = np.max(data)
        return max_value
    
    def channel_rms(self, index):
        data = self.recording.get_data(picks=[index])
        rms_value = np.sqrt(np.mean(np.square(data)))
        return rms_value

    def combined_signals(self, plot=False):
        data = self.recording.get_data()

        # Fourier Transform each signal and sum them up along the epochs and channels axis
        combined_fourier_transform = np.sum(np.fft.fft(data, axis=2), axis=(0, 1))

        # Average the combined Fourier Transform (if needed)
        combined_fourier_transform /= (data.shape[0] * data.shape[1])

        # Inverse Fourier Transform to get the final combined signal
        final_combined_signal = np.fft.ifft(combined_fourier_transform)
        if plot: 
            # Plotting the final combined signal
            plt.plot(self.recording.times, final_combined_signal)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Final Combined Signal')
            plt.show()
        return final_combined_signal
    
    def spectral_data(self, index=0):
        # Suppress verbose messages by setting the logging level to 'WARNING'
        logging.basicConfig(level=logging.WARNING)
        set_log_level('WARNING')
        signal = self.__getitem__(0)
        fs = 250
        # Define the frequency range
        fmin = 1
        fmax = 40

        # Define common frequency grid
        freq_grid = np.linspace(fmin, fmax, num=fmax)

        # Periodogram
        frequencies_p, psd_p = periodogram(signal, fs)
        # Select the frequencies between fmin and fmax
        indices_p = np.where((frequencies_p >= fmin) & (frequencies_p <= fmax))
        feature_vector_p = psd_p[indices_p]
        # Interpolate PSD values to the common grid for Periodogram
        feature_vector_p = np.interp(freq_grid, frequencies_p[indices_p], feature_vector_p)

        # Welch
        frequencies_w, psd_w = welch(signal, fs, nperseg=fs)
        # Select the frequencies between fmin and fmax
        indices_w = np.where((frequencies_w >= fmin) & (frequencies_w <= fmax))
        feature_vector_w = psd_w[indices_w]

        # Multitaper
        frequencies_m, psd_m = psd_array_multitaper(signal, fs, fmin=fmin, fmax=fmax, adaptive=True)
        # Check if psd_m has the second dimension
        if len(psd_m.shape) > 1:
            feature_vector_m = psd_m[0]
        else:
            feature_vector_m = psd_m
        # Interpolate PSD values to the common grid for Multitaper
        feature_vector_m = np.interp(freq_grid, frequencies_m, feature_vector_m)

        return np.array([feature_vector_p, feature_vector_w, feature_vector_m]).reshape(-1)
    
    def get_periodogram(self, index=0):
        # Suppress verbose messages by setting the logging level to 'WARNING'
        logging.basicConfig(level=logging.WARNING)
        set_log_level('WARNING')
        signal = self.__getitem__(0)
        fs = 250
        # Define the frequency range
        fmin = 1
        fmax = 40
        # Define common frequency grid
        freq_grid = np.linspace(fmin, fmax, num=fmax)

        # Periodogram
        frequencies_p, psd_p = periodogram(signal, fs)
        # Select the frequencies between fmin and fmax
        indices_p = np.where((frequencies_p >= fmin) & (frequencies_p <= fmax))
        feature_vector_p = psd_p[indices_p]
        # Interpolate PSD values to the common grid for Periodogram
        feature_vector_p = np.interp(freq_grid, frequencies_p[indices_p], feature_vector_p)
        return feature_vector_p
    
    def get_welch(self, index=0):
        # Suppress verbose messages by setting the logging level to 'WARNING'
        logging.basicConfig(level=logging.WARNING)
        set_log_level('WARNING')
        signal = self.__getitem__(0)
        fs = 250
        # Define the frequency range
        fmin = 1
        fmax = 40
        # Define common frequency grid
        freq_grid = np.linspace(fmin, fmax, num=fmax)
        # Welch
        frequencies_w, psd_w = welch(signal, fs, nperseg=fs)
        # Select the frequencies between fmin and fmax
        indices_w = np.where((frequencies_w >= fmin) & (frequencies_w <= fmax))
        feature_vector_w = psd_w[indices_w]
        return feature_vector_w
    
    def get_welch_feature(self, index=0):
        mean = self.channel_mean(index)
        variance = self.channel_variance(index)
        iqr = self.channel_iqr(index)
        #welch_mean = self.get_welch(index).mean()
        #return np.array([mean, variance, iqr])
        return self.get_welch(index)
        
    def get_multitaper(self, index=0):
        logging.basicConfig(level=logging.WARNING)
        set_log_level('WARNING')
        signal = self.__getitem__(0)
        fs = 250
        # Define the frequency range
        fmin = 1
        fmax = 40
        # Define common frequency grid
        freq_grid = np.linspace(fmin, fmax, num=40)
        # Multitaper
        frequencies_m, psd_m = psd_array_multitaper(signal, fs, fmin=fmin, fmax=fmax, adaptive=True)
        # Check if psd_m has the second dimension
        if len(psd_m.shape) > 1:
            feature_vector_m = psd_m[0]
        else:
            feature_vector_m = psd_m
        # Interpolate PSD values to the common grid for Multitaper
        feature_vector_m = np.interp(freq_grid, frequencies_m, feature_vector_m)
        return feature_vector_m
    
    
    def getmne_multitaper(self, index=0):
        signal = self.__getitem__(index)
        fs = self.recording.info['sfreq']
        
        multitapered = mne.time_frequency.psd_array_multitaper(signal, sfreq=fs, fmin=1, fmax=40)
        data = multitapered[0]
        # Assuming the original data is in the frequency range of 1 to N
        original_freqs = np.linspace(1, len(data), len(data))

        # Desired frequency range and size
        desired_freqs = np.linspace(1, 40, 40)

        # Interpolation using numpy
        interpolated_data_np = np.interp(desired_freqs, original_freqs, data)
        
        return interpolated_data_np
    
    def getmne_periodogram(self , index=0):
        signal = self.__getitem__(index)
        fs = self.recording.info['sfreq']
        
        if len(signal.shape) == 1:
            signal = signal[np.newaxis, :]
        
        # Compute the PSD using Welch's method
        psd, freqs = mne.time_frequency.psd_array_welch(
            signal, sfreq=fs, fmin=1, fmax=40, 
            n_fft=1024, n_overlap=256, n_per_seg=512  # Adjusted parameters
        )
        
        # Desired frequency range and size
        desired_freqs = np.linspace(1, 40, 40)
        
        # Interpolation using numpy
        interpolated_psd = np.interp(desired_freqs, freqs, psd[0])
        
        return interpolated_psd
        
    def getmne_welch(self , index=0):
        signal = self.__getitem__(index)
        fs = self.recording.info['sfreq']
        
        if len(signal.shape) == 1:
            signal = signal[np.newaxis, :]
        
        
        # Define the window length and noverlap based on your requirements
        window_length = signal.shape[1] // 4  # 1/4 of the EEG signal length
        noverlap = window_length // 2  # half the window length
        
        # Compute the PSD using Welch's method
        psd, freqs = mne.time_frequency.psd_array_welch(
            signal, sfreq=fs, fmin=1, fmax=40, 
            n_fft=window_length, n_overlap=noverlap, n_per_seg=window_length
        )
        
        # Desired frequency range and size
        desired_freqs = np.linspace(1, 40, 40)
        
        # Interpolation using numpy
        interpolated_psd = np.interp(desired_freqs, freqs, psd[0])
        
        return interpolated_psd

def getmne_welch(signal, fs=250):
    # Ensure the signal is 2D (n_channels, n_times)
    if len(signal.shape) == 1:
        signal = signal[np.newaxis, :]
    
    # Define the window length and noverlap based on your requirements
    window_length = signal.shape[1] // 4  # 1/4 of the EEG signal length
    noverlap = window_length // 2  # half the window length
    
    # Compute the PSD using Welch's method
    psd, freqs = mne.time_frequency.psd_array_welch(
        signal, sfreq=fs, fmin=1, fmax=40, 
        n_fft=window_length, n_overlap=noverlap, n_per_seg=window_length
    )
    
    # Desired frequency range and size
    desired_freqs = np.linspace(1, 40, 40)
    
    # Interpolation using numpy
    interpolated_psd = np.interp(desired_freqs, freqs, psd[0])
    
    return interpolated_psd
    
    return interpolated_data_np
    
    
    
    
    def plot_bands(self, signal=None, fs = 250, bands={
            'Delta (0.1-4 Hz)': (0.1, 4),  # Modified lower bound
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-12 Hz)': (8, 12),
            'Beta (12-30 Hz)': (12, 30),
            'Gamma (30-45 Hz)': (30, 45)} ):
        def bandpass_filter(data, lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return lfilter(b, a, data)
        if signal == None: 
            signal = self.combined_signals()
        
        t = np.linspace(0, signal.shape[0]/fs, signal.shape[0], endpoint=False)
        
        # Extract and plot each frequency band
        plt.figure(figsize=(12, 8))
        plt.subplot(len(bands) + 1, 1, 1)
        plt.plot(t, signal)
        plt.title('Original Signal')

        for i, (band_name, (low, high)) in enumerate(bands.items(), start=2):
            filtered_signal = bandpass_filter(signal, low, high, fs)
            mean_amplitude = np.mean(np.abs(filtered_signal))

            plt.subplot(len(bands) + 1, 1, i)
            plt.plot(t, filtered_signal)
            plt.title(f'{band_name} - Mean Amplitude: {mean_amplitude}')
            

        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

    def get_channel_array(self):
        return self.recording.get_data()
    
    

class SubjectType():
    def __init__(self):
        self.alzheimer = []
        self.ftd = [] 
        self.normal = []
        
        
    def shuffle(self):
        random.shuffle(self.alzheimer)
        random.shuffle(self.ftd)
        random.shuffle(self.normal)

class ExperimentDataset():
    def __init__(self, data, label, makeCategories=None):
        self.data = data
        self.label = label
        if makeCategories != None:
            self.ad_cn = ExperimentDataset.classSpliter(data, label, incl=['A','C'])
            self.ad_ftd = ExperimentDataset.classSpliter(data, label, incl=['A','F'])
            self.cn_ftd = ExperimentDataset.classSpliter(data, label, incl=['C','F'])
        
    @staticmethod
    def classSpliter(data, labels, incl=['A', 'C']):
        newData = []
        newLabel = []
        for i in range(0, len(data)):
            if labels[i] in incl:
                newData.append(data[i])
                newLabel.append(labels[i])
        return ExperimentDataset(newData, newLabel)
        
    def get_distribution(self):
        label_counts = Counter(self.label)
        total_samples = len(self.label)

        # Sort the unique labels alphabetically
        sorted_labels = sorted(label_counts.keys())

        # Print the sorted unique labels, their counts, and percentages
        for label in sorted_labels:
            count = label_counts[label]
            percentage = (count / total_samples) * 100
            print(f"{label} --> {count} :--> {percentage:.2f}%")
            
    @staticmethod
    def get_distribution_label(label):
        label_counts = Counter(label)
        total_samples = len(label)

        # Sort the unique labels alphabetically
        sorted_labels = sorted(label_counts.keys())

        # Print the sorted unique labels, their counts, and percentages
        for label in sorted_labels:
            count = label_counts[label]
            percentage = (count / total_samples) * 100
            print(f"{label} --> {count} :--> {percentage:.2f}%")






class Experiment():
    def __init__(self, data):
        labels = []
        for epoch in data: 
            labels.append(epoch.label)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)

        
        self.train = ExperimentDataset(X_train, y_train, makeCategories=True)
        self.test = ExperimentDataset(X_test,y_test, makeCategories=True)
        
        
        
        
        
         
    
    


class EEGDataset(Dataset):
    def __init__(self, root, epoch_length=2.0, overlap=1.0):
        """
        Initializes an EEGDataset instance.
        
        Parameters:
        - root: The root directory where the dataset is located.
        - epoch_length: Length (in seconds) of each epoch. Default is 2.0 seconds.
        - overlap: Overlap (in seconds) between consecutive epochs. Default is 1.0 second.
        """
        self.epochs = []
        self.subjects = []
        
        self.type = SubjectType()
        
        # Load subjects using DS004504 class
        ds = DS004504(root)
        bbb = 0
        
        for participant in tqdm(ds):
            # Create a Subject object
            subject = Subject(
                participantId=participant['participant_id'],
                gender=participant['Gender'],
                age=participant['Age'],
                group=participant['Group'],
                mmse=participant['MMSE'],
                raw_path=participant['raw_path'],
                filtered_path=participant['preprocessed_path']
            )
            self.subjects.append(subject)

            # Load the raw recording
            bbb += 1 
            #print(f"{bbb} - {participant['Group']} - {subject.get_filtered_path()}")
            
            with HiddenPrints():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_eeglab(subject.get_filtered_path())
                epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, preload=True, overlap=overlap)
                
                
                for i in range(len(epochs)): 
                    epoch = Epoch(subject, subject.get_filtered_path(),  recording=epochs[i])
                    self.epochs.append(epoch)
                    
                    if epoch.label == 'A':
                        self.type.alzheimer.append(epoch)
                    elif epoch.label == 'F':
                        self.type.ftd.append(epoch)
                    elif epoch.label == 'C':
                        self.type.normal.append(epoch)                    
                    else:
                        ...

        self.experiments = Experiment(self.epochs)

    def __len__(self):
        """
        Returns the total number of epochs in the dataset.
        """
        return len(self.epochs)

    def __getitem__(self, index):
        """
        Fetches the EEG data and label for a given epoch index.
        
        Parameters:
        - index: Index of the epoch to fetch.

        Returns:
        - Tuple of (epoch_data, label): Where epoch_data is the EEG data for the given epoch and label is its associated label.
        """
        epoch = self.epochs[index]
        
        if hasattr(epoch, 'recording'):
            return epoch
        else: 
            epoch.getEpochRecording()
            return epoch
    
    
    
    
    def shuffle(self):
        random.shuffle(self.epochs)
        
        
    def getStatisticalFeatures(self):
        channelFrames = {}
        sample_epoch = self.epochs[0]
        n_channels = len(sample_epoch.recording.info['ch_names'])
        
        # Define the column names for the DataFrames
        columns = ['Mean', 'Median', 'Mode', 'Variance', 'StdDev', 'Skewness', 'Kurtosis', 'IQR', 'Entropy', 'Min', 'Max', 'RMS', 'Label']
        
        for index in range(n_channels):
            for channel in sample_epoch.recording.info['ch_names']:
                # Initialize each DataFrame with the defined column names
                channelFrames[channel] = pd.DataFrame(columns=columns)
        with HiddenPrints():
            warnings.filterwarnings("ignore", category=FutureWarning)
            for epoch in tqdm(self.epochs):
                for i, name in enumerate(epoch.recording.info['ch_names']):
                    # Create a dictionary to store the statistical features for the current channel
                    channel_data = {
                        'Mean': epoch.channel_mean(i),
                        'Median': epoch.channel_median(i),
                        'Mode': epoch.channel_mode(i),
                        'Variance': epoch.channel_variance(i),
                        'StdDev': epoch.channel_std_dev(i),
                        'Skewness': epoch.channel_skewness(i),
                        'Kurtosis': epoch.channel_kurtosis(i),
                        'IQR': epoch.channel_iqr(i),
                        'Entropy': epoch.channel_entropy(i),
                        'Min': epoch.channel_min(i),
                        'Max': epoch.channel_max(i),
                        'RMS': epoch.channel_rms(i),
                        'Label': epoch.label  # Assuming 'epoch' has a 'label' attribute to identify the label of the epoch
                    }

                    # Append the data for the current channel and epoch to the respective DataFrame
                    channelFrames[name].loc[len(channelFrames[name])] = channel_data

        return channelFrames
        
        
        




def ML_EEG(model, XX_train, yy_train, XX_test, yy_test, label="lda"):
    X = np.array(XX_train)
    y = np.array(yy_train)

    # Initialize LDA object
    lda = model

    # Initialize StratifiedKFold object
    cv = StratifiedKFold(n_splits=10)

    # Initialize metrics
    accuracies = []
    sensitivities = []
    specificities = []
    precisions = []
    f1_scores = []

    for train_index, test_index in cv.split(X, y):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Sensitivity (recall)
        sens = recall_score(y_test, y_pred, average='macro')
        sensitivities.append(sens)

        # Specificity
        spec = (cm.sum(axis=1) - np.diag(cm)) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
        specificities.append(np.mean(spec))  # Taking mean of specificity for all classes

        # Precision
        prec = precision_score(y_test, y_pred, average='macro')
        precisions.append(prec)

        # F1-score
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)

    # Calculate average metrics
    average_accuracy = np.mean(accuracies)
    average_sensitivity = np.mean(sensitivities)
    average_specificity = np.mean(specificities)
    average_precision = np.mean(precisions)
    average_f1_score = np.mean(f1_scores)

    print(f"{label}")
    # Print average metrics
    print(f"Average Model Accuracy: {average_accuracy * 100:.2f}%")
    print(f"Average Sensitivity/Recall: {average_sensitivity:.2f}")
    print(f"Average Specificity: {average_specificity:.2f}")
    print(f"Average Precision: {average_precision:.2f}")
    print(f"Average F1 Score: {average_f1_score:.2f}")  
    
    print(f"& {average_sensitivity*100:.1f} & {average_specificity*100:.1f} & {average_precision*100:.1f} & {average_f1_score*100:.1f} & {average_accuracy*100:.1f}")
    y_pred_final = lda.predict(XX_test)
    y_test_final = yy_test

    # Calculate and print metrics for the final test set
    accuracy_final = accuracy_score(y_test_final, y_pred_final)
    sensitivity_final = recall_score(y_test_final, y_pred_final, average='weighted')
    precision_final = precision_score(y_test_final, y_pred_final, average='weighted')
    f1_score_final = f1_score(y_test_final, y_pred_final, average='weighted')

    # Calculating Specificity for the model as a whole can be complex and might require a one-vs-all approach for multi-class problems.
    # However, for binary classification, it can be calculated from the confusion matrix.
    cm_final = confusion_matrix(y_test_final, y_pred_final)
    specificity_final = cm_final[1,1] / (cm_final[1,1] + cm_final[0,1])  # For binary classification

    #print(f"\nValidation Test Results:")
    #print(f"Accuracy: {accuracy_final * 100:.2f}%")
    #print(f"Sensitivity/Recall: {sensitivity_final:.2f}")
    #print(f"Specificity: {specificity_final:.2f}")
    #print(f"Precision: {precision_final:.2f}")
    #print(f"F1 Score: {f1_score_final:.2f}")   
        
        
        
        
