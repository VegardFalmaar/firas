from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt     # type: ignore


def load():
    file = Path('firas_monopole_spec_v1.txt')
    data = np.loadtxt(file)
    return data.T


def main():
    frequency, monopole_spectrum, res_monopole_spectrum, spectrum_uncertainty, galaxy_spectrum = load()
    print(frequency)
    print(monopole_spectrum)
    print(res_monopole_spectrum)
    print(spectrum_uncertainty)
    print(galaxy_spectrum)

    fig, ax = plt.subplots()
    ax.plot(frequency, monopole_spectrum)
    fig.savefig('results/spectrum.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
