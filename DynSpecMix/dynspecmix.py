"""
Load sets of simulated stellar spectra for quiet/spot/faculae regions,
and generate dynamic spectra via time-dependent mixtures.

Created 2024-05-10 by Tom Loredo
"""
from pathlib import Path

import numpy as np
from numpy import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

import h5py

# To simplify dependencies, just copy in SpectraSet, below.
# from ssspec import SpectraSet


# Larger plot labels:
rc('axes', labelsize=18)
rc('xtick.major', pad=8)
rc('xtick', labelsize=14)
rc('ytick.major', pad=8)
rc('ytick', labelsize=14)
rc('figure.subplot', bottom=.19, top=.925)


def plot_full_cont(sset):
    """
    Plot the full and continuum emergent intensities for a SpectraSet with
    quiet/spot/faculae components.
    """
    fig, ax = subplots(figsize=(9,7))
    xlabel('$\\lambda$ (\u00c5)')
    ylabel('Emergent intensity')

    lambdas = sset.lambdas
    plot(lambdas, sset.full_q, '-', alpha=.7, label='Full (quiet)')
    plot(lambdas, sset.cont_q, '-', alpha=.7, label='Cont. (quiet)')
    plot(lambdas, sset.full_s, '-', alpha=.7, label='Full (spot)')
    plot(lambdas, sset.cont_s, '-', alpha=.7, label='Cont. (spot)')
    plot(lambdas, sset.full_f, '-', alpha=.7, label='Full (fac.)')
    plot(lambdas, sset.cont_f, '-', alpha=.7, label='Cont. (fac.)')
    legend()


def plot_rel(sset):
    """
    Plot the relative (full/continuum) spectra for a SpectraSet with
    quiet/spot/faculae components.
    """
    fig, ax = subplots(figsize=(9,7))
    xlabel('$\\lambda$ (\u00c5)')
    ylabel('Relative intensity')

    lambdas = sset.lambdas
    plot(lambdas, sset.full_q/sset.cont_q, '-', alpha=.7, label='Quiet')
    plot(lambdas, sset.full_s/sset.cont_s, '-', alpha=.7, label='Spot')
    plot(lambdas, sset.full_f/sset.cont_f, '-', alpha=.7, label='Faculae')
    legend()


# This is copied from ssspec.py in the simple stellar spectra package.
# Be sure to propagate any changes! 

class SpectraSet:
    """
    Manage a set of related spectra, handling storage to and loading from an
    HDF5 store.
    """

    def __init__(self, h5_path=None, lambdas=None):
        """
        Initialize a collection of spectra, either loading existing spectra 
        from an HDF5 store if `h5_path` is provided, or preparing to save and
        store spectra on a common wavelength grid, `lambdas`, if no path is
        provided.
        """
        self.names = []
        if h5_path is not None:
            self.h5_path = Path(h5_path)

            with h5py.File(h5_path, 'r') as store:
                print('Loaded HDF5 store {}:'.format(h5_path))

                # Load all items in the HDF5 store into the namespace.
                for name, value in store.items():
                    print('  {}: {}'.format(name, value))
                    setattr(self, name, value)
                    self.names.append(name)

                # Load all attrs (metadata) into the namespace.
                for name, value in store.attrs.items():
                    print('  {}: {}'.format(name, value))
                    setattr(self, name, value)
                    self.names.append(name)

            self.n_w = len(self.lambdas)
        else:
            self.lambdas = lambdas

    def __setattr__(self, name, value):
        if name in ('names', 'h5_path', 'n_w'):  # expected, unsaved attributes
            object.__setattr__(self, name, value)
        else:  # all other attributes should be array-like
            self.names.append(name)
            object.__setattr__(self, name, asarray(value))

    def save(self, h5_path, **meta):
        self.h5_path = Path(h5_path)
        if self.h5_path.exists():
            raise ValueError('A file exists at the provided path!')
 
        with h5py.File(h5_path, 'w') as store:

            for name, value in meta.items():
                store.attrs[name] = value

            for name in self.names:
                store.create_dataset(name, data=getattr(self, name),
                    compression='gzip')


class SpecMixer:
    """
    Abstract base class for building SpectraSet mixers.
    """

    def __init__(self, sset):
        self.sset = sset
        self.n_w = sset.n_w

    def full_cont(self, t):
        """
        Compute the full and continuum spectra at time `t`.
        """
        wt_spot = self.mod_spot(t)
        wt_fac = self.mod_fac(t)
        wt_quiet = 1. - wt_spot - wt_fac

        full = wt_quiet*self.sset.full_q + wt_spot*self.sset.full_s \
            + wt_fac*self.sset.full_f
        cont = wt_quiet*self.sset.cont_q + wt_spot*self.sset.cont_s \
            + wt_fac*self.sset.cont_f
        return full, cont

    def dynspec(self, tvals):
        """
        Compute arrays containing the dynamic spectrum, with row index
        corresponding to times in `tvals`, and column index corresponding
        to the wavelengths of the component spectra.
        """
        full = empty((len(tvals), self.n_w))
        cont = empty((len(tvals), self.n_w))
        for i, t in enumerate(tvals):
            full[i], cont[i] = self.full_cont(t)
        return full, cont
