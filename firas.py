from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt     # type: ignore
import scipy.constants as const     # type: ignore


# Pre-calculate the prefactor 2 h / c^2 for the Planck BB spectrum.
# To accept frequency in units of cm^-1, this prefactor is multiplied with
# (100c)^3 making it 2e6 h c.
# Finally, to convert into MJy / sr it is
# multiplied with 1e-6 and 1e26, giving 2e26 h c.
PLANCK_BB_FACTOR_1 = 2e26 * const.Planck * const.c

# Pre-calculate the factor h / k in the exponential of the Planck BB spectrum.
# To accept frequency in units of cm^-1, it is multiplied with 100c making it
# 100 h c / k.
PLANCK_BB_FACTOR_2 = 100 * const.Planck * const.c / const.Boltzmann


def load():
    file = Path('firas_monopole_spec_v1.txt')
    return np.loadtxt(file).T


def planck_bb(frequency: float, temperature: float):
    """Calculate the Planck blackbody radiation spectrum.

    args:
        frequency (float, ArrayLike[float]):
            The frequencies at which the Planck blackbody spectrum should be
            calculated. Should have units cm^-1.
        temperature (float): The temperature of the blackbody in units of K.
    returns:
        (float, ArrayLike[float]):
            The calculated spectrum in units of MJy / sr.
    """
    factor = PLANCK_BB_FACTOR_1 * frequency**3
    return factor / (np.exp(PLANCK_BB_FACTOR_2*frequency/temperature) - 1)


def main():
    frequencies, monopole_spectrum, res_monopole_spectrum, spectrum_uncertainty, galaxy_spectrum = load()

    planck = planck_bb(frequencies, temperature=2.725)

    fig, ax = plt.subplots()
    ax.plot(
        frequencies,
        planck,
        label='2.275 K Planck BB'
    )
    ax.scatter(
        frequencies,
        monopole_spectrum,
        marker='+',
        color='r',
        label='Spectrum'
    )
    ax.set_xlabel('Frequency, cm^-1')
    ax.set_ylabel('Spectrum, MJy / sr')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('results/spectrum.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
