import numpy as np
from scipy.signal import hilbert

def compute_wsmi(signal1, signal2, num_bins=4, delay=1):
    """
    Compute Weighted Symbolic Mutual Information (wSMI) between two signals.
    Args:
        signal1 (numpy array): First EEG signal.
        signal2 (numpy array): Second EEG signal.
        num_bins (int): Number of bins for symbolic encoding.
        delay (int): Delay parameter for embedding.
    Returns:
        float: Weighted Symbolic Mutual Information.
    """
    def symbolic_encoding(signal, num_bins):
        """Symbolically encode the signal into discrete states."""
        bins = np.linspace(np.min(signal), np.max(signal), num_bins + 1)
        encoded = np.digitize(signal, bins) - 1  # Ensure indices start at 0
        encoded[encoded == num_bins] = num_bins - 1  # Handle edge case
        return encoded

    def compute_joint_distribution(encoded1, encoded2):
        """Compute joint probability distribution of two encoded signals."""
        joint_hist = np.zeros((num_bins, num_bins))
        for x, y in zip(encoded1, encoded2):
            joint_hist[x, y] += 1
        joint_hist /= np.sum(joint_hist)
        return joint_hist

    # Symbolic encoding
    encoded1 = symbolic_encoding(signal1[::delay], num_bins)
    encoded2 = symbolic_encoding(signal2[::delay], num_bins)

    # Joint and marginal probabilities
    joint_prob = compute_joint_distribution(encoded1, encoded2)
    marg_prob1 = np.sum(joint_prob, axis=1)
    marg_prob2 = np.sum(joint_prob, axis=0)

    # Compute wSMI
    wsmi = 0
    for i in range(num_bins):
        for j in range(num_bins):
            if joint_prob[i, j] > 0:
                wsmi += joint_prob[i, j] * np.log2(
                    joint_prob[i, j] / (marg_prob1[i] * marg_prob2[j])
                )
    return wsmi

def compute_wpli(signal1, signal2):
    """
    Compute Weighted Phase Lag Index (wPLI) between two signals.
    Args:
        signal1 (numpy array): First EEG signal.
        signal2 (numpy array): Second EEG signal.
    Returns:
        float: Weighted Phase Lag Index.
    """
    # Compute the analytic signals using the Hilbert transform
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Compute the instantaneous phase difference
    phase_diff = np.angle(analytic_signal1) - np.angle(analytic_signal2)

    # Compute the imaginary component of the phase difference
    imag_phase_diff = np.imag(np.exp(1j * phase_diff))

    # Compute wPLI
    numerator = np.sum(np.abs(imag_phase_diff) * np.sign(imag_phase_diff))
    denominator = np.sum(np.abs(imag_phase_diff))
    wpli = np.abs(numerator / denominator) if denominator != 0 else 0
    return wpli


# Example Usage
if __name__ == "__main__":
    # Simulated EEG signals (replace with actual EEG data)
    eeg_signal1 = np.random.normal(0, 1, 1000)
    eeg_signal2 = np.random.normal(0, 1, 1000)

    # Compute wSMI
    wsmi_value = compute_wsmi(eeg_signal1, eeg_signal2, num_bins=4, delay=1)
    print(f"Weighted Symbolic Mutual Information (wSMI): {wsmi_value:.4f}")

    # Compute wPLI
    wpli_value = compute_wpli(eeg_signal1, eeg_signal2)
    print(f"Weighted Phase Lag Index (wPLI): {wpli_value:.4f}")