import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from itertools import combinations

def bandpass_filter(data, low, high, fs):
    """
    Apply a bandpass filter to EEG data.
    Args:
        data (numpy array): EEG signal.
        low (float): Low cutoff frequency.
        high (float): High cutoff frequency.
        fs (float): Sampling frequency.
    Returns:
        numpy array: Filtered signal.
    """
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(2, [low, high], btype="band")
    return filtfilt(b, a, data)

def symbolic_transform(data, k, tau):
    """
    Transform EEG signal into symbols based on permutation entropy.
    Args:
        data (numpy array): EEG signal.
        k (int): Length of the symbol (number of points per symbol).
        tau (int): Temporal separation between points in the symbol.
    Returns:
        numpy array: Symbolic representation of the signal.
    """
    n = len(data)
    symbols = []
    for i in range(n - (k - 1) * tau):
        subsequence = data[i:i + k * tau:tau]
        rank = np.argsort(subsequence)
        symbols.append(tuple(rank))
    return np.array(symbols)

def compute_weighted_smi(signal1, signal2, k=3, tau=1):
    """
    Compute Weighted Symbolic Mutual Information (wSMI) between two EEG signals.
    Args:
        signal1, signal2 (numpy arrays): EEG signals.
        k (int): Length of the symbol (default=3).
        tau (int): Temporal separation parameter (default=1).
    Returns:
        float: Weighted Symbolic Mutual Information.
    """
    symbols1 = symbolic_transform(signal1, k, tau)
    symbols2 = symbolic_transform(signal2, k, tau)
    unique_symbols1, counts1 = np.unique(symbols1, axis=0, return_counts=True)
    unique_symbols2, counts2 = np.unique(symbols2, axis=0, return_counts=True)
    joint_symbols, joint_counts = np.unique(
        np.stack((symbols1, symbols2), axis=1), axis=0, return_counts=True
    )

    # Compute probabilities
    p1 = counts1 / np.sum(counts1)
    p2 = counts2 / np.sum(counts2)
    p_joint = joint_counts / np.sum(joint_counts)

    # Compute wSMI
    wsmi = 0
    for p in p_joint:
        if p > 0:
            wsmi += p * np.log2(p / (p1 * p2))
    return wsmi

def compute_wpli(signal1, signal2, fs):
    """
    Compute Weighted Phase Lag Index (wPLI) between two EEG signals.
    Args:
        signal1, signal2 (numpy arrays): EEG signals.
        fs (float): Sampling frequency.
    Returns:
        float: Weighted Phase Lag Index.
    """
    # Hilbert transform to get analytic signal
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Compute phase differences
    phase_diff = np.angle(analytic_signal1) - np.angle(analytic_signal2)

    # Compute imaginary part of phase differences
    imag_phase_diff = np.imag(np.exp(1j * phase_diff))

    # Compute wPLI
    num = np.abs(np.mean(imag_phase_diff))
    den = np.mean(np.abs(imag_phase_diff))
    wpli = num / den if den != 0 else 0
    return wpli

def compute_connectivity_measures(eeg_data, fs, frequency_bands, k=3, tau=1):
    """
    Compute wSMI and wPLI for all electrode pairs across all frequency bands.
    Args:
        eeg_data (numpy array): EEG data (channels x time).
        fs (float): Sampling frequency.
        frequency_bands (list of tuples): List of (low, high) cutoff frequencies.
        k (int): Length of the symbol for wSMI (default=3).
        tau (int): Temporal separation parameter for wSMI (default=1).
    Returns:
        dict: Connectivity measures (wSMI and wPLI) for each frequency band.
    """
    num_channels = eeg_data.shape[0]
    results = {band: {"wSMI": [], "wPLI": []} for band in frequency_bands}

    for band in frequency_bands:
        low, high = band
        filtered_data = np.array([bandpass_filter(eeg_data[i], low, high, fs) for i in range(num_channels)])

        for i, j in combinations(range(num_channels), 2):
            signal1, signal2 = filtered_data[i], filtered_data[j]

            # Compute wSMI
            wsmi_value = compute_weighted_smi(signal1, signal2, k, tau)

            # Compute wPLI
            wpli_value = compute_wpli(signal1, signal2, fs)

            # Store results
            results[band]["wSMI"].append(wsmi_value)
            results[band]["wPLI"].append(wpli_value)

    return results

# Example EEG data (channels x time)
eeg_data = np.random.rand(64, 1250)  # 64 channels, 5 seconds at 250 Hz
fs = 250  # Sampling frequency
frequency_bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]  # Delta, Theta, Alpha, Beta

# Compute connectivity measures
results = compute_connectivity_measures(eeg_data, fs, frequency_bands)

# Example output
for band in frequency_bands:
    print(f"Frequency Band: {band}")
    print("wSMI:", results[band]["wSMI"])
    print("wPLI:", results[band]["wPLI"])