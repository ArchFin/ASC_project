import numpy as np

def binarise_signal(signal, threshold=None):
    """
    Binarise a continuous signal around a threshold (default: mean of the signal).
    Args:
        signal (numpy array): Continuous signal.
        threshold (float): Threshold for binarisation. Defaults to the mean of the signal.
    Returns:
        str: Binarised signal as a string of '0's and '1's.
    """
    if threshold is None:
        threshold = np.mean(signal)
    binary_signal = (signal > threshold).astype(int)
    return ''.join(binary_signal.astype(str))

def compute_lzc(binary_sequence):
    """
    Compute Lempel-Ziv Complexity for a binary sequence using the formula:
        C(y_n) = k log2(n) / n
    Args:
        binary_sequence (str): Binary string representing the signal.
    Returns:
        float: Normalised Lempel-Ziv Complexity (C(y_n)).
    """
    n = len(binary_sequence)
    k = 1  # Number of unique patterns
    i = 0  # Pointer in the sequence
    sub_seq = binary_sequence[i]

    # Main LZC computation
    while i + len(sub_seq) < n:
        if binary_sequence.startswith(sub_seq, i + len(sub_seq)):
            sub_seq += binary_sequence[i + len(sub_seq)]
        else:
            k += 1
            i += len(sub_seq)
            sub_seq = binary_sequence[i]

    # Normalised Lempel-Ziv Complexity
    lzc = (k * np.log2(n)) / n
    return lzc

def compute_lzc_for_eeg(eeg_data):
    """
    Compute the Lempel-Ziv Complexity for EEG data.
    Args:
        eeg_data (numpy array): 1D EEG signal.
    Returns:
        float: Normalised Lempel-Ziv Complexity.
    """
    # Step 1: Binarise the EEG signal
    binary_sequence = binarise_signal(eeg_data)

    # Step 2: Compute LZC
    lzc = compute_lzc(binary_sequence)
    return lzc

# Example Usage
if __name__ == "__main__":
    # Simulated EEG data (replace with actual EEG data)
    eeg_data = np.random.normal(0, 1, 1000)  # Example random EEG-like signal

    # Compute LZC
    lzc_value = compute_lzc_for_eeg(eeg_data)
    print(f"Normalised Lempel-Ziv Complexity (C(y_n)): {lzc_value:.4f}")

import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to the data.
    Args:
        data (numpy array): The signal to filter.
        cutoff (float): The cutoff frequency.
        fs (float): Sampling rate of the signal.
        order (int): Order of the filter.
    Returns:
        numpy array: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def compute_rank_vectors(signal, k, tau):
    """
    Compute rank vectors for the signal based on subsequence sorting.
    Args:
        signal (numpy array): Input signal.
        k (int): Length of each subsequence.
        tau (int): Time lag between elements in subsequences.
    Returns:
        list: List of rank vectors.
    """
    n = len(signal)
    rank_vectors = []
    for i in range(n - (k - 1) * tau):
        subsequence = signal[i : i + k * tau : tau]
        rank_vector = tuple(np.argsort(subsequence))
        rank_vectors.append(rank_vector)
    return rank_vectors

def compute_bplzc(rank_vectors):
    """
    Compute the BPLZC by compressing rank vectors into unique, non-repeating subsequences.
    Args:
        rank_vectors (list): List of rank vectors.
    Returns:
        float: Normalised BPLZC.
    """
    n = len(rank_vectors)
    k = 1  # Number of unique patterns
    i = 0  # Pointer in the sequence
    sub_seq = [rank_vectors[i]]

    # Main BPLZC computation
    while i + len(sub_seq) < n:
        if rank_vectors[i + len(sub_seq)] in sub_seq:
            sub_seq.append(rank_vectors[i + len(sub_seq)])
        else:
            k += 1
            i += len(sub_seq)
            sub_seq = [rank_vectors[i]]

    # Normalised BPLZC
    bplzc = (k * np.log2(n)) / n
    return bplzc

def compute_bplzc_for_eeg(eeg_data, sampling_rate, k=3, tau=1):
    """
    Compute the Bandt Pompe Lempel Ziv Complexity for EEG data.
    Args:
        eeg_data (numpy array): 1D EEG signal.
        sampling_rate (float): Sampling rate of the EEG signal.
        k (int): Length of each subsequence (default: 3).
        tau (int): Time lag between elements in subsequences (default: 1).
    Returns:
        float: Normalised BPLZC.
    """
    # Step 1: Apply a low-pass filter to the EEG data
    cutoff_frequency = sampling_rate / (tau * k)
    filtered_signal = butter_lowpass_filter(eeg_data, cutoff_frequency, sampling_rate)

    # Step 2: Compute rank vectors
    rank_vectors = compute_rank_vectors(filtered_signal, k, tau)

    # Step 3: Compute BPLZC
    bplzc = compute_bplzc(rank_vectors)
    return bplzc

# Example Usage
if __name__ == "__main__":
    # Simulated EEG data (replace with actual EEG data)
    eeg_data = np.random.normal(0, 1, 1000)  # Example random EEG-like signal
    sampling_rate = 250  # Sampling rate in Hz

    # Compute BPLZC with default parameters K=3, τ=1
    bplzc_value = compute_bplzc_for_eeg(eeg_data, sampling_rate, k=3, tau=1)
    print(f"Bandt Pompe Lempel Ziv Complexity (BPLZC): {bplzc_value:.4f}")