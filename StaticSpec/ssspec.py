"""
Simple Stellar Spectrum

A simple model of a stellar spectrum with absorption lines, for testing
spectrum analysis algorithms.

More accurately, this offers a *crude* and simple stellar spectrum!

Main simplifying assumptions:
* Plane-parallel atmosphere
* LTE line description

Possible directions for improving (or at least complicating) the model:

* Add direction dependence to account for limb darkening
* Add more atmosphere models, particularly for active regions (spots, plages...)
* Add depth dependence to line opacity, either ad hoc (based on tabulated
  formation depths), or crude calculations using atmosphere properties
* Add alternatives to the skew normal for profile asymmetry, e.g., via
  continuous location mixtures
* Add time-dependent atmosphere capability
* Add velocity gradient capability (see NSO Collage labs)

Created 2024-04-13 by Tom Loredo
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
from numpy import *
import scipy
from scipy.special import wofz, voigt_profile

import h5py

from astropy import constants as const
from astropy.modeling.physical_models import BlackBody

from asym_voigt import SkewVoigt


# Constants in CGS units:
c = const.c.cgs.value
h = const.h.cgs.value
k_B = const.k_B.cgs.value
amu = 1.6605e-24  # atomic mass unit
hc = h*c
h2_cc = 2.*h/c**2
h2cc = 2.*h*c**2
ang2cm = 1.e-8
h2cc_ang = ang2cm*h2cc  # for radiance per angstrom

# Atomic masses for elements w/ Fraunhofer lines (am_C = 12 by def'n, for C12)
# [from PubChem, https://pubchem.ncbi.nlm.nih.gov/ptable/atomic-mass/]:
amasses = dict(C=12.011, H=1.0080, He=4.0026, N=14.007, O=15.999, Na=22.9898, 
    Mg=24.305, Si=28.085, Ca=40.078, Fe=55.845, Ni=58.693, Hg=200.59)

# TODO:  What is the impact of isotope diversity on the Doppler width?
# Kurucz 1995 notes isotopes lead to blends that get mistaken for lines with
# excessive broadening.  Is there some adjustment to mass for the Doppler
# shift that is optimal, perhaps motivated by properties of discrete mixtures?


def doppler_std(am, T, nu):
    """
    Doppler width for a line associated with a species of atomic mass `am`
    at temperature `T` (K), with (nominal) line frequency `nu` (Hz).

    The width is the standard deviation of the thermal distribution of
    Doppler shifts.
    """
    return sqrt(k_B*T/(am*amu*c**2)) * nu


class BBSpecRad:
    """
    Black body spectral radiance.
    """

    def __init__(self, temp):
        """
        Initialize by specifying the black body temperature in degrees K.
        """
        self.temp = temp
        self.kT = k_B*temp

    def rad_f(self, nu):
        """
        Black body spectral radiance per unit frequency, vs. frequency in Hz,
        for temperature `temp` (K).

        Units are erg / (s cm^2 sr Hz).
        """
        nu = asarray(nu)
        arg = h*nu/self.kT
        return h2_cc*nu**3/expm1(arg)

    def rad_w(self, lam):
        """
        Black body spectral radiance per unit wavelength, vs. wavelength in
        angstroms, for temperature `temp` (K)

        Units are erg / (s cm^2 sr ang).
        """
        lam = asarray(lam)*ang2cm  # wavelength in cm
        arg = hc/(lam*self.kT)
        return h2cc_ang/(lam**5*expm1(arg))


# TODO:  Is there a role for caching Voigt profiles, e.g., at zero location,
# and using shifted versions?  Would require lines sharing other params.


class SpectralLine:
    """
    Define and implement a spectral line profile for radiative transfer
    calculations.
    """

    def __init__(self, amass, ang=None, nm=None, hz=None, wt=1.,
        fwhm_f=None, fwhm_w=None, v_t=0., asym=0., name=None):
        """
        Define a spectral line for a species with atomic mass `amass`.

        The line center may be specified via *one* of:

            ang, wavelength in angstroms
            nm, wavelength in nanometers
            hz, frequency in Hz

        `wt` is the weight coefficient (multiplying the PDF to give the
        profile); it is unity by default.

        The FWHM parameter for the Lorentzian component of the line profile
        (e.g., from the natural width, possibly including pressure broadening)
        may be specified via *one* of:

            fwhm_f, FWHM in frequency
            fwhm_w, approx. FWHM in wavelength (angstroms)

        An asymmetry parameter, `asym`, is optional.  Note that the line
        center will be very near but not exactly at the mode if asym != 0.

        Note that, internally, profile calculations are done as a function of
        frequency, and transformed to wavelength when necessary.
        """
        self.amass = amass
        self.mass = amass*amu
        self.wt = wt
        self.v_t = v_t
        self.asym = asym
        self.name = name

        # TODO:  Probably should require wavelength in angstroms (or nm),
        # since those units also impact FWHM.  Or specify units separately.

        if ang is not None:
            if nm is not None or hz is not None:
                raise ValueError('Provide only one of ang/nm/hz!')
            self.ang = ang
            self.nm = ang/10.
            self.hz = c/(self.ang*ang2cm)
        elif nm is not None:
            if hz is not None:
                raise ValueError('Provide only one of ang/nm/hz!')
            self.nm = nm
            self.ang = 10.*nm
            self.hz = c/(self.ang*ang2cm)
        else:
            if hz is None:
                raise ValueError('Provide one of ang/nm/hz!')
            self.hz = hz
            self.ang = (c/hz)/ang2cm
            self.nm = self.ang/10.

        self.set_gamma(fwhm_f=fwhm_f, fwhm_w=fwhm_w)

    def set_gamma(self, fwhm_f=None, fwhm_w=None):
        if fwhm_f is not None:
            if fwhm_w is not None:
                raise ValueError('Provide only one of fwhm_f/fwhm_w!')
            self.fwhm_f = fwhm_f
            self.fwhm_w = self.ang * fwhm_f * (self.ang*ang2cm/c)
        else:
            if fwhm_w is None:
                raise ValueError('Provide one of fwhm_f/fwhm_w!')
            self.fwhm_w = fwhm_w
            self.fwhm_f = c*fwhm_w/self.ang/(self.ang*ang2cm)

    def profile_f(self, nu, T_dop, fwhm_f=None, v_t=None, asym=None, pdf=True):
        """
        Compute the line profile as a function of frequency for frequencies
        `nu`, using a thermal Doppler component with nominal (symmetric) width 
        specified by temperature `T_dop` (K).

        The Lorentzian FWHM may be optionally specified as `fwhm_f`, overriding
        a previously set value, and similarly for the turbulent velocity,
        `v_t`, and the asymmetry parameter, `asym`.

        If `pdf` is True, the profile will be evaluated as a PDF scaled by
        the `wt` attribute (so with area equal to `wt`).  Otherwise, the 
        profile will be computed to have an amplitude of `wt` at the line
        center.
        """
        nu = asarray(nu)
        if fwhm_f is not None:
            self.set_gamma(fwhm_f)
        if v_t is not None:
            self.v_t = v_t
        if asym is not None:
            self.asym = asym

        sig_dop = sqrt(k_B*T_dop/(self.mass*c**2))  # thermal Doppler factor
        sig_dop = sqrt(sig_dop**2 + (self.v_t/c)**2) * self.hz
        gamma = 0.5*self.fwhm_f
        # print('widths for', self.name, sig_dop, gamma)

        if self.asym == 0.:
            profile = self.wt * voigt_profile(nu-self.hz, sig_dop, gamma)
        else:
            svoigt = SkewVoigt(self.hz, self.fwhm_f, sig_dop, self.asym)
            profile = self.wt * svoigt.profile(nu)

        if pdf:
            return profile
        else:
            if self.asym == 0.:
                ampl = voigt_profile([0.], sig_dop, gamma)[0]
            else:
                ampl = svoigt.profile([0.])[0]
            return profile / ampl

    def profile_w(self, ang, T_dop, fwhm_w=None, v_t=None, asym=None, pdf=True):
        """
        Compute the line profile as a function of wavelength in angstroms,
        `ang`, using a thermal Doppler component with nominal (symmetric) width 
        specified by temperature `T_dop` (K).

        The Lorentzian FWHM may be optionally specified as `fwhm_w`, overriding
        a previously set value, and similarly for the turbulent velocity,
        `v_t`, and the asymmetry parameter, `asym`.

        The profile is computed as a density vs. frequency, and transformed 
        to a density vs. wavelength (and rescaled as appropriate).
        """
        ang = asarray(ang)
        nu = c/(ang*ang2cm)
        if fwhm_w is not None:
            self.set_gamma(fwhm_w=fwhm_w)
        if v_t is not None:
            self.v_t = v_t
        if asym is not None:
            self.asym = asym

        # Adjust using the Jacobian between frequency and wavelength:
        profile = self.profile_f(nu, T_dop, pdf=pdf) * c/(ang*ang2cm)**2
        if not pdf:
            profile *= (self.ang*ang2cm)**2 / c
        return profile


def emergent_intens(lambdas, atmos, lines=[], log=False):
    """
    Compute the emergent intensity by solving the RTE on the wavelength grid
    `lambdas` (angstroms), for an LTE atmosphere specified by `atmos`, and
    a list of lines in `lines`.
    """

    # Atmosphere info, bottom to top:
    T = flip(atmos.T)
    tau_c = flip(10.**atmos.log_tau_c)
    n_int = len(T) - 1  # number of quadrature intervals

    # Initial conditions --- source function and tau at the bottom.
    source = BBSpecRad(T[0]).rad_w(lambdas)
    intens = source
    phi = zeros_like(lambdas)
    for line in lines:
        phi += line.profile_w(lambdas, T[0], pdf=False)
    # Total optical depth vs. wavelength, treating line weights as
    # relative to the continuum:
    tau_w = tau_c[0]*(1. + phi)


    # Integrate over the atmosphere using the trapezoid rule.
    phi_n = empty_like(lambdas)  # avoid memory alloc'n for each iter
    for i in range(1, n_int+1):
 
        # Source function at next T:
        source_n = BBSpecRad(T[i]).rad_w(lambdas)

        # Compute the line profile for the next T.
        phi_n[:] = 0.
        for line in lines:
            if log:
                if (i+1)%log == 0:
                    print('  Line:', line.name)
            phi_n += line.profile_w(lambdas, T[i], pdf=False)

        # Total optical depth vs. wavelength for next tau:
        tau_w_n = tau_c[i]*(1. + phi_n)

        # Trapezoid rule, for each wavelength:
        dtau = -(tau_w_n - tau_w)
        exp_tau = exp(-dtau)
        intens = intens*exp_tau + 0.5*dtau*(source + source_n*exp_tau)

        source = source_n
        tau_w = tau_w_n

        if log:
            if (i+1)%log == 0:
                print('Step {} of {}...'.format(i+1, n_int))
                print(dtau.min(), dtau.max())

    return intens


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




# Solar photosphere effective temperature, from:
# Photosphere - Wikipedia
# https://en.wikipedia.org/wiki/Photosphere#Temperature
T_quiet = 5772.
bb_sol = BBSpecRad(T_quiet)

# Typical sunspot umbral temperatures span 3000 - 4500 K; see:
# Sunspot - Wikipedia
# https://en.wikipedia.org/wiki/Sunspot
T_spot = 3750.
bb_spot = BBSpecRad(T_spot)



# Load data from the FAL C solar model of Fontenla+ (1993).
# FAL93 model C is for an "average intensity area" of the quiet Sun.
raw = loadtxt("falc.dat", unpack=True, skiprows=1)

SolarAtmos = namedtuple('SolarAtmos', ['log_tau_c', 'h', 'T', 'p_g', 'p_e', 'n'])

FAL_C = SolarAtmos(raw[0], raw[1], raw[2], raw[3], raw[4], len(raw[1]))
atmos = FAL_C

# A *really* crude spot atmosphere model, simply scaling down T:
atmos_spot = atmos._replace(T=(T_spot/T_quiet)*atmos.T)

# A *really* crude faculum atmosphere model, simply increasing T:
atmos_fac = atmos._replace(T=atmos.T+300.)


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import *

    from asym_voigt import fwhm2sig, lorentzian, gauss_fwhm

    try:
        import myplot
        from myplot import ipy_ion, close_all, csavefig
        ipy_ion()
        #myplot.tex_on()
        csavefig.save = False
    except ImportError:
        ion()


    # Full wavelength & frequency ranges of interest:
    lam_ll, lam_uu = 2000, 7000
    nu_ll, nu_uu = c/lam_uu/ang2cm, c/lam_ll/ang2cm

    # Calcium line data:

    # Wavelengths, frequencies:
    lam_H, lam_K = 3968.47, 3933.66  # Wikipedia; vacuum wavelengths?
    nu_H, nu_K = c/lam_H/ang2cm, c/lam_K/ang2cm

    # Wavelength, freq. ranges spanning Ca H & K lines:
    lam_l, lam_u = 3910., 4010.
    nu_l, nu_u = c/lam_u/ang2cm, c/lam_l/ang2cm

    # 1 K Doppler widths (std dev'n); width grows like sqrt(T):
    am_Ca = amasses['Ca']
    w_dop_1_H = sqrt(k_B/(am_Ca*amu*c**2)) * nu_H
    w_dop_1_K = sqrt(k_B/(am_Ca*amu*c**2)) * nu_K

    wdop_H = w_dop_1_H*sqrt(atmos.T)
    wdop_K = w_dop_1_K*sqrt(atmos.T)

    Gamma_damping = 1.e7  # Milic's value for Fe


    # BB spectra, first vs. freq:
    bbfig = figure(figsize=(14,10))
    subplots_adjust(hspace=.3, wspace=.25)

    subplot(221)
    xlabel('Frequency (Hz)')
    ylabel('BB spectral radiance (per Hz)')
    nus = linspace(nu_ll, nu_uu, 1001)
    plot(nus, bb_sol.rad_f(nus), label='Solar')
    plot(nus, bb_spot.rad_f(nus), label='Spot')
    legend()

    # Vs. wavelength, using BB scale units to do conversion:
    subplot(222)
    xlabel(r'$\lambda$ (ang)')
    ylabel(r'BB spectral radiance (per $\AA$)')
    lams = linspace(lam_ll, lam_uu, 1001)
    plot(lams, bb_sol.rad_w(lams), label='Solar')
    plot(lams, bb_spot.rad_w(lams), label='Spot')
    legend()

    # BB(freq) converted to wavelength:
    subplot(223)
    xlabel(r'$\lambda$ (ang)')
    ylabel(r'BB spectral radiance (per $\AA$)')
    jac = c/(lams*ang2cm)**2  # for radiance per cm wavelength
    jac = ang2cm*jac  # convert to per angstrom
    plot(lams, jac*bb_sol.rad_f(c/lams/ang2cm), label='Solar')
    plot(lams, jac*bb_spot.rad_f(c/lams/ang2cm), label='Spot')
    legend()


    # Parameters for Doppler (Gaussian) and Cauchy factors.
    fwhm_v, fwhm_p = 0.2, .05
    sig = fwhm2sig(fwhm_v)
    gamma = fwhm_p/2.


    # Skew Voigt example:
    svfig = figure(figsize=(9,7))
    axvline(0., ls='-', c='k')

    skew = 2.
    sv = SkewVoigt(0., fwhm_p, sig, skew)

    x = np.linspace(-0.9,0.9,1000)

    plt.plot(x, lorentzian(x, 0., fwhm_p), ls='--', label='Lorentzian')
    plt.plot(x, gauss_fwhm(x, 0., fwhm_v), ls=':', lw=3, alpha=.4, label='Gaussian')
    plt.plot(x, sv.snorm.pdf(x), ls=':', label='Skew norm')
    plt.plot(x, sv.profile(x), c='b', ls='-', lw=1, alpha=.6, label='SkewVoigt')

    plt.legend()


    # Plot the solar atmosphere model.
    sfig = figure(figsize=(14,10))
    subplots_adjust(hspace=.3, wspace=.25)

    subplot(221)
    xlabel('$h$ (km)')
    ylabel('$T$ (K)')
    plot(FAL_C.h/1.e5, FAL_C.T, ls='-', lw=2, label='Quiet')
    plot(atmos_spot.h/1.e5, atmos_spot.T, ls='-', lw=2, label='Spot')
    legend()

    subplot(222)
    xlabel('$h$ (km)')
    ylabel(r'$\log\tau$')
    plot(FAL_C.h/1.e5, FAL_C.log_tau_c, ls='-', lw=2)

    subplot(223)
    xlabel(r'$\log\tau$')
    ylabel('$T$ (K)')
    plot(FAL_C.log_tau_c, FAL_C.T, ls='-', lw=2)
    l, u = xlim()
    xlim(u, l)

    subplot(224)
    xlabel(r'$\tau$')
    ylabel('$T$ (K)')
    plot(10.**FAL_C.log_tau_c, FAL_C.T, ls='-', lw=2)
    l, u = xlim()
    xlim(u, l)


    # Doppler widths:
    wfig = figure(figsize=(7,5))
    plot(atmos.T, wdop_H/nu_H, '-', lw=2, label='Ca H')
    plot(atmos.T, wdop_K/nu_K, '-', lw=2, label='Ca H')
    xlabel('$T$ (K)')
    ylabel('Doppler width (relative to line frequency)')
    legend()

