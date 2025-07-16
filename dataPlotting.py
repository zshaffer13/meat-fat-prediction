from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ramanspy import Spectrum

def plot_spectra_by_target(spectral_data: Union[pd.DataFrame, Spectrum],
                           spectral_axis: Union[np.ndarray, list],
                           meta_data: pd.DataFrame,
                           target: str = 'fat',
                           alpha: float = 0.6,
                           cmap=plt.cm.viridis):
    """
    Plot spectra colored by a target variable.
    
    Parameters:
    - spectral_data: DataFrame or ramanspy Spectrum (rows = samples)
    - spectral_axis: list or array of wavenumbers/wavelengths
    - meta_data: DataFrame with metadata (same number of rows)
    - target: which metadata column to color by
    - alpha: transparency of lines
    - cmap: matplotlib colormap
    """
    if target not in meta_data.columns:
        raise ValueError(f"'{target}' not found in metadata columns")

    if isinstance(spectral_data, pd.DataFrame):
        spectra = spectral_data.values
    elif isinstance(spectral_data, Spectrum):
        spectra = spectral_data.spectral_data
    else:
        raise TypeError("spectral_data must be a pandas DataFrame or ramanspy Spectrum")

    values = meta_data[target]
    norm = plt.Normalize(values.min(), values.max())

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(spectra)):
        ax.plot(spectral_axis, spectra[i], color=cmap(norm(values.iloc[i])), alpha=alpha)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f'{target.capitalize()}')

    ax.set_xlabel("Wavelength / Wavenumber")
    ax.set_ylabel("Absorbance")
    ax.set_title(f"Spectra Colored by {target.capitalize()}")
    plt.tight_layout()
    plt.show()

    

def plot_mean_spectra_by_group(spectral_data: Union[pd.DataFrame, Spectrum],
                               spectral_axis: Union[np.ndarray, list],
                               meta_data: pd.DataFrame,
                               target: str = 'fat',
                               n_bins: int = 3,
                               bin_labels: list = None,
                               title: str = None):
    """
    Plot mean spectra for quantile-binned groups of a target variable.
    """

    if target not in meta_data.columns:
        raise ValueError(f"'{target}' not found in metadata columns")

    if bin_labels is None:
        bin_labels = [f"Group {i+1}" for i in range(n_bins)]

    # Bin the target variable
    binned = pd.qcut(meta_data[target], q=n_bins, labels=bin_labels)

    # Extract spectra
    if isinstance(spectral_data, pd.DataFrame):
        spectra = spectral_data.values
    elif isinstance(spectral_data, Spectrum):
        spectra = spectral_data.spectral_data
    else:
        raise TypeError("spectral_data must be a pandas DataFrame or ramanspy Spectrum")

    plt.figure(figsize=(10, 6))
    for label in binned.unique():
        indices = binned == label
        mean_spectrum = np.mean(spectra[indices], axis=0)
        plt.plot(spectral_axis, mean_spectrum, label=f"{label}")

    plt.legend()
    plt.xlabel("Wavelength / Wavenumber")
    plt.ylabel("Mean Absorbance")
    if title is None:
        title = f"Mean Spectra by {target.capitalize()} Group"
    plt.title(title)
    plt.tight_layout()
    plt.show()



