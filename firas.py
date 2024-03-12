from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt     # type: ignore
import scipy.constants as const     # type: ignore

import plots

plots.use_tex()


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
    """Load the FIRAS data.

    returns:
        (np.2darray of shape (5, 43)): The FIRAS data.
            - Column 1: frequencies, units = cm^-1
            - Column 2: FIRAS monopole spectrum, units = MJy/sr
            - Column 3: residual monopole spectrum, units = MJy/sr
            - Column 4: spectrum uncertainty (1-sigma), units = MJy/sr
            - Column 5: modeled Galaxy spectrum at the Galactic poles,
                units = MJy/sr
    """
    file = Path('firas_monopole_spec_v1.txt')
    result = np.loadtxt(file).T
    # convert columns 3-5 from kJy/sr to MJy/sr
    result[2:] *= 1e-3
    return result


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


def plot_spectrum(temperature: float):
    frequencies, monopole_spectrum, *_ = load()

    planck = planck_bb(frequencies, temperature)

    fig, ax = plt.subplots()
    ax.plot(
        frequencies,
        planck,
        label=f'{temperature:.3f} K Planck BB'
    )
    ax.scatter(
        frequencies,
        monopole_spectrum,
        marker='+',
        color='r',
        label='Spectrum'
    )
    plots.set_ax_info(
        ax,
        xlabel='Frequency, cm$^{-1}$',
        ylabel='Spectrum, MJy / sr',
    )
    ax.grid(True)
    fig.tight_layout()
    file_id = f'{temperature:.3f}'.replace('.', '_')
    fig.savefig(f'results/spectrum-{file_id}.pdf')
    plt.close(fig)


def chi_squared(observed, expected, sigma) -> float:
    diff = observed - expected
    return np.sum(diff**2 / sigma**2)


def plot_chi_squared_with_temperature():
    frequencies, monopole_spectrum, _, sigma, _ = load()

    temperatures = np.linspace(2.6, 2.8, 11)
    chi_sq = np.zeros_like(temperatures)

    for i, temperature in enumerate(temperatures):
        plot_spectrum(temperature)
        planck = planck_bb(frequencies, temperature)
        chi_sq[i] = chi_squared(monopole_spectrum, planck, sigma)

    fig, ax = plt.subplots()
    ax.plot(temperatures, chi_sq)
    plots.set_ax_info(
        ax,
        xlabel='Temperature, K',
        ylabel=r'$\chi^2$',
        legend=False,
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('results/chi-squared.pdf')
    plt.close(fig)


def find_best_fit():
    frequencies, monopole_spectrum, _, sigma, _ = load()

    T = 2.7
    dT = 0.0001

    current = chi_squared(monopole_spectrum, planck_bb(frequencies, T), sigma)
    nxt = chi_squared(monopole_spectrum, planck_bb(frequencies, T + dT), sigma)

    while nxt < current:
        current = nxt
        T += dT
        nxt = chi_squared(
            monopole_spectrum,
            planck_bb(frequencies, T + dT),
            sigma
        )
    print(f'Best fit temperature: {T:.4f} K')
    print(f'Corresponding chi-squared: {current:.4e}')


def main():
    plot_spectrum(2.725)
    plot_chi_squared_with_temperature()
    find_best_fit()


if __name__ == '__main__':
    main()
