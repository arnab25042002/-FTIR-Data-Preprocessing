import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

def load_ftir_data(file_path):
    """Load FTIR data from a CSV file."""
    data = pd.read_csv(file_path)
    return data['Wavenumber'], data['Absorbance']

def smooth_data(absorbance, window_length=25, polyorder=3):
    """Apply Savitzky-Golay filter to smooth the data."""
    return savgol_filter(absorbance, window_length=window_length, polyorder=polyorder)

def detect_peaks(smoothed_absorbance, wavenumber):
    """Detect peaks based on zero crossings and confirm with find_peaks."""
    second_derivative = np.gradient(np.gradient(smoothed_absorbance, wavenumber), wavenumber)
    zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]
    peak_indices, _ = find_peaks(smoothed_absorbance)
    selected_peaks = [i for i in zero_crossings if i in peak_indices]
    return selected_peaks

def filter_peaks_by_range(peaks, wavenumber, smoothed_absorbance, ranges):
    """Filter peaks that fall within specified ranges and retain the highest in each range."""
    filtered_peaks = []
    for start, end in ranges:
        peaks_in_range = [idx for idx in peaks if start <= wavenumber[idx] <= end]
        if peaks_in_range:
            highest_peak = max(peaks_in_range, key=lambda idx: smoothed_absorbance[idx])
            filtered_peaks.append(highest_peak)
    return filtered_peaks

def plot_spectrum(wavenumber, smoothed_absorbance, filtered_peaks, ranges):
    """Plot the FTIR spectrum with annotated peaks."""
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumber, smoothed_absorbance, label='Smoothed Spectrum (Savitzky-Golay)', color='black')
    
    # Highlight ranges
    for start, end in ranges:
        plt.axvspan(start, end, color='yellow', alpha=0.2)

    # Annotate peaks
    for peak_idx in filtered_peaks:
        plt.annotate(f'{wavenumber[peak_idx]:.0f}',
                     (wavenumber[peak_idx], smoothed_absorbance[peak_idx]),
                     textcoords="offset points", xytext=(0, 10), ha='center',
                     fontsize=9, color='blue')

    plt.gca().invert_xaxis()  # FTIR spectra usually have the x-axis inverted
    plt.title('FTIR Spectrum of Sample 3')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorbance')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main Script
file_path = r"C:\Users\ARNAB BANDYOPADHYAY\Downloads\sample 3.csv"
wavenumber, absorbance = load_ftir_data(file_path)

# User-specified ranges for peak detection
specified_ranges = [(1600, 1700), (1750, 1850), (1200, 1350),(2700,2880),(2881,3000),(3300,3400),(3450,3550)]  # Modify as needed

smoothed_absorbance = smooth_data(absorbance)
peaks = detect_peaks(smoothed_absorbance, wavenumber)
filtered_peaks = filter_peaks_by_range(peaks, wavenumber, smoothed_absorbance, specified_ranges)
plot_spectrum(wavenumber, smoothed_absorbance, filtered_peaks, specified_ranges)
