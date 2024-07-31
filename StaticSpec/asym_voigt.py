"""
Approximate asymmetric voigt line profile functions.

Experimentation shows that the Gauss-Hermite approach used in some Voigt
function papers fails badly when the natural/pressure-broadened line width
is narrow compared to the Doppler width.  The Lorentzians at the nodes
are narrow and appear like spikes in the result.

Created 2024-02-21 by Tom Loredo
"""

import numpy as np
from numpy import *
import scipy

from scipy.special import roots_hermite
from scipy.stats import cauchy, norm, skewnorm
from scipy.special import wofz, voigt_profile
from scipy.integrate import quad

from gauss_trapz import GaussTrap_12_5


# Constants for GHQuad:
rtpi = sqrt(pi)
rt2 = sqrt(2)

# Constants for Voigt function calculations:
irt2 = 1./sqrt(2.)
lg2 = log(2.)
rt2pi = sqrt(2*pi)
rt2_pi = sqrt(2/pi)
rtpi_2 = sqrt(pi)/2.
rtlg2 = sqrt(lg2)
rtlg2_pi = sqrt(lg2/pi)

# For converting b/t FWHM and standard deviation:
sig2fwhm_ = 2.*sqrt(2.*lg2)


def sig2fwhm(sig):
    """
    Return the FWHM for a Gaussian PDF with standard deviation `sig`.
    """
    return sig2fwhm_*sig

def fwhm2sig(fwhm):
    """
    Return the standard deviation for a Gaussian PDF with FWHM `fwhm`.
    """
    return fwhm/sig2fwhm_


class GHQuad:
    """
    Gauss-Hermite quadrature.
    """

    def __init__(self, mu, sig, n):
        """
        Set up a Gauss-Hermite quadrature rule for integrating f(x)*normal(x)
        for a normal with mean `mu` and std devn `sig` using quadrature with
        `n` nodes.
        """
        self.mu = mu
        self.sig = sig
        self.n = n

        # roots_hermite integrates with exp(-x**2) wt func, absorbing the wt.
        # Shift and scale to y = (x-mu)/(sig*rt2); multiply result by
        # by sig*rt2, which cancels the 1/(sig*rt2pi) factor in a normalized
        # Gaussian PDF weight function.

        # Abscissas & weights absorbing the Gaussian wt func:
        absc, wts = roots_hermite(n)
        self.absc = mu + rt2*sig*absc
        self.wts = wts/rtpi 

    def quad(self, f, *args):
        """
        Return the quadrature of f(x)*normal(x), i.e., `f` is the full
        integrand *divided by a normal PDF*.
        """
        self.fvals = f(self.absc, *args)
        return sum(self.wts*self.fvals)


# 3 functions from: The Voigt profile
# https://scipython.com/book2/chapter-8-scipy/examples/the-voigt-profile/


def gauss_fwhm(x, x_c, fwhm):
    """
    Return Gaussian line shape at x with center `x_c`, FWHM `fwhm`.
    """
    alpha = 0.5*fwhm  # HWHM
    return rtlg2_pi / alpha * np.exp(-((x - x_c) / alpha)**2 * lg2)

def lorentzian(x, x_c, fwhm):
    """
    Return Lorentzian line shape at x with FWHM `fwhm`.
    """
    gamma = 0.5*fwhm  # HWHM
    return 1. / (((x -x_c)/gamma)**2 + 1.) / (gamma*pi)

def voigt_wofz(x, x_c, fwhm_v, fwhm_p):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.
    """
    sigma = fwhm_v / sig2fwhm_
    gamma = 0.5*fwhm_p
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

def voigt_sig_fwhm(x, x_c, sigma, fwhm):
    """
    Return the Voigt line shape at `x` for a profile with center `x_c`,
    Gaussian component standard deviation `sigma`, and Lorentzian component
    FWHM `fwhm`.
    """
    gamma = 0.5*fwhm
    return voigt_profile(x-x_c, sigma, gamma)

def voigt_fwhm(x, x_c, fwhm_v, fwhm_p):
    """
    Return the Voigt line shape at x with Lorentzian component FWHM gamma
    and Gaussian component FWHM alpha.
    """
    sigma = fwhm_v / sig2fwhm_
    gamma = 0.5*fwhm_p
    return voigt_profile(x-x_c, sigma, gamma)

def voigt_hwhm(x, x_c, hwhm_v, hwhm_p):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.
    """
    sigma = 2.*hwhm_v / sig2fwhm_
    gamma = hwhm_p
    return voigt_profile(x-x_c, sigma, gamma)

def voigt2pt(x, x_c, fwhm_v, fwhm_p):
    """
    Pseudo-Voigt profile using 2-pt Gauss-Hermite quadrature.
    """
    sigma = fwhm_v / sig2fwhm_
    gamma = 0.5*fwhm_p
    # dx = irt2*fwhm_v/rtlg2/pi
    dx = sigma
    return (lorentzian(x, x_c-dx, fwhm_p) + lorentzian(x, x_c+dx, fwhm_p)) / 2.



def voigt_gh(x, x_c, fwhm_v, fwhm_p, gh):
    """
    Pseudo-Voigt profile using Gauss-Hermite quadrature.
    """
    sigma = fwhm_v / sig2fwhm_
    gamma = 0.5*fwhm_p
    # dx = irt2*fwhm_v/rtlg2/pi
    v = 0.
    for dx, w in zip(gh.absc, gh.wts):
        v += w*lorentzian(x, dx+x_c, fwhm_p)
    return v


class CauchyConvolution:
    """
    Cauchy-weighted convolution via quadrature for unimodal integrands,
    using a coordinate transformation and composite hybrid Gauss-trapezoid
    quadrature for the convolution.

    Devised for computing generalized Voigt profiles, this computes a
    standard Voigt profile when the integrand is a Gaussian (i.e., the
    Doppler frequency offset distribution).
    """

    def __init__(self, x_c, fwhm):
        """
        Initialized a quadrature instance using a Cauchy weight factor
        with center `x_c` and FWHM `fwhm`.
        """
        self.x_c = x_c
        self.fwhm = fwhm
        self.hwhm = 0.5*fwhm
        self.cauchy = cauchy(loc=self.x_c, scale=self.hwhm)

    def convolve(self, x, f, f_core, quad, xtra=False, log=False):
        """
        Compute the convolution integral of the function `f` times the
        base Cauchy PDF at offset(s) `x`, using quadrature method `quad`
        applied to the core range of `f` given as `f_core` (in x units),
        and to each side of the core (in y space).
        """
        x = asarray(x)
        vals = empty_like(x)
        x_l, x_u = f_core

        def f_y(y):
            x_y = xx - self.cauchy.ppf(y)
            return f(x_y)

        for i, xx in enumerate(x):
            # First the core, noting that x and y grow in inverted directions.
            y_c_l = self.cauchy.cdf(xx - x_u)
            y_c_u = self.cauchy.cdf(xx - x_l)
            vals[i] = quad(y_c_l, y_c_u, f_y)
            # Add the bounding regions, to y=0 and y=1.
            vals[i] += quad(0., y_c_l, f_y)
            vals[i] += quad(y_c_u, 1., f_y)
        if not xtra:
            return vals
        else:  # note these are just the last values
            return vals, y_c_l, y_c_u

    def vintgnd_y(self, y, x, sig):
        """
        Integrand for transformed Voigt integral, as a function of
        `y` in [0.1], for given `x` in [-inf, inf], with Doppler
        std dev'n `sig`.
        """
        arg = x - self.cauchy.ppf(y)
        return exp(-0.5*(arg/sig)**2)/(sig*rt2pi)


class SkewVoigt:
    """
    An asymmetric generalization of the Voigt profile, replacing the normal
    (Doppler) factor in the convolution with a skew normal.
    """

    def __init__(self, nu_c, fwhm_w, sig_dop, skew_dop):
        """
        Initialize a skew Voigt profile with nominal line center `nu_c` and 
        FWHM for the Lorentzian wings `fwhm_w` (corresponding to natural width,
        pressure broadening, etc.), and with a skewed normal Doppler core
        component with standard deviation `sig_dop` and skewness `skew_dop`.

        The skew normal parameterization makes the line profile mode close to
        `nu_c`.
        """
        self.nu_c = nu_c
        self.fwhm_w = fwhm_w
        self.sig_dop = sig_dop
        self.skew_dop = skew_dop

        self.cconv = CauchyConvolution(nu_c, fwhm_w)
        self.gtquad = GaussTrap_12_5(3)

        # Derived parameters for skew normal:
        dd = skew_dop**2/(1 + skew_dop**2)
        self.scl_dop = sig_dop/sqrt(1. - 2.*dd/pi)
        dd_dop = rt2_pi*sqrt(dd)
        if skew_dop == 0.:
            self.loc_dop = 0.
            self.snorm = norm(loc=self.loc_dop, scale=self.scl_dop)
        else:
            # Use a "quite accurate" approx. mode that is ~10% off for skew ~ 1:
            self.loc_dop = -dd + (1.-pi/4) * dd**3 / (1.-dd**2) + \
                0.5*sign(skew_dop)*exp(-2*pi/abs(skew_dop))
            self.loc_dop = self.scl_dop*self.loc_dop
            self.snorm = skewnorm(skew_dop, loc=self.loc_dop, scale=self.scl_dop)

            # TODO:  Refine this via minimize? May introduce discontinuity.
            # Perhaps a fixed number of Newton-Raphson steps would work.
            # The skew norm *mean* is avbl in closed form; would it be the final
            # profile mean?  The latter probably does not exist due to tails.
            # But maybe the underlying snorm mean would be a good location 
            # parameter.

        # Frequency range for the core of the skew norm dist'n:
        self.f_core = (self.snorm.ppf(.01), self.snorm.ppf(.99))

    def profile(self, nu):
        """
        Skew Voigt profile as a function of frequency.
        """
        nu = asarray(nu)
        return self.cconv.convolve(nu, self.snorm.pdf,
            self.f_core, self.gtquad.quad)


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import *
    from scipy.optimize import minimize

    try:
        import myplot
        from myplot import ipy_ion, close_all, csavefig
        ipy_ion()
        #myplot.tex_on()
        csavefig.save = False
    except ImportError:
        ion()

    # Parameters for Doppler (Gaussian) and Cauchy factors.
    fwhm_v, fwhm_p = 0.2, .05
    sig = fwhm2sig(fwhm_v)
    gamma = fwhm_p/2.

    # Doppler integrand for Voigt:
    n = norm(0., sig)
    f = n.pdf  # integrand factor

    # x = np.linspace(-0.6,0.6,1000)
    x = np.linspace(-0.9,0.9,1000)

    # Voigt case, showing integrand factors and the final profile, for comparison:
    vfig = figure(figsize=(9,7))
    axvline(0., ls='-', c='k')

    plt.plot(x, lorentzian(x, 0., fwhm_p), ls='--', label='Lorentzian')
    plt.plot(x, gauss_fwhm(x, 0., fwhm_v), ls=':', label='Gaussian')
    # plt.plot(x, voigt_wofz(x, 0., fwhm_v, fwhm_p), label='Voigt (wolz)')
    plt.plot(x, voigt_fwhm(x, 0., fwhm_v, fwhm_p), ls='-', lw=2, label='Voigt')
    # plt.plot(x, voigt2pt(x, 0., fwhm_v, fwhm_p), ls='--', alpha=.5, label='2pt GH')

    # A common Voigt approximatin, using Gauss-Hermite quadrature:
    gh = GHQuad(0., sig, 9)
    plt.plot(x, voigt_gh(x, 0., fwhm_v, fwhm_p, gh), ls='--', lw=2, label='9pt GH')

    # Cauchy convolution to reproduce Voigt:
    # Use a normal approximation to identify the "core" quadrature range.
    fwhm_t = .5*fwhm_p
    fwhm_t = fwhm_t + sqrt(fwhm_t**2 + fwhm_v**2)
    print('FWHMs:', fwhm_v, fwhm_p, fwhm_t)
    sig_t = fwhm2sig(fwhm_t)
    n_t = norm(0., sig_t)
    nqf = n_t.ppf
    f_core = (nqf(.01), nqf(.99))

    cc = CauchyConvolution(0., fwhm_p)
    gt_3 = GaussTrap_12_5(3)
    plt.plot(x, cc.convolve(x, f, f_core, gt_3.quad), c='k', ls='--', lw=2, label='CC 12_5')

    # Do the transformed quadrature via SciPy's quad.
    xvals = []
    qvals = []
    for i, xx in enumerate(x[::20]):
        # eps of 1.e-5 produces very slight dips around 0.82.
        q, err, info = quad(cc.vintgnd_y, 0., 1., (xx, sig),
            epsabs=1.e-6, epsrel=1.e-6, full_output=True)
        xvals.append(xx)
        qvals.append(q)
        if i % 10 == 0:
            print(i, xx, q, err, info['neval'])
    plt.plot(xvals, qvals, 'bD', ms=4, alpha=.5, label='quad')

    plt.legend()


    # Asymmetric case, using a skew normal Doppler dist'n:
    afig = figure(figsize=(9,7))
    axvline(0., ls='-', c='k')


    def skewnorm_mode(skew):
        if skew == 0.:
            return 0.
        d = skew/sqrt(1+skew**2)
        dd = rt2_pi*d
        return dd - (1.-pi/4) * dd**3 / (1.-dd**2) - 0.5*sign(skew)*exp(-2*pi/abs(skew))

    skew = 1.
    # loc = - rt2_pi*sig*skew/sqrt(1+skew**2)
    loc = -sig*skewnorm_mode(skew)
    # Note sig here is the scale parameter, equal to std dev'n only for skew=0.
    snorm = skewnorm(skew, loc=loc, scale=sig)

    plt.plot(x, lorentzian(x, 0., fwhm_p), ls='--', label='Lorentzian')
    plt.plot(x, gauss_fwhm(x, 0., fwhm_v), ls=':', lw=3, alpha=.4, label='Gaussian')
    plt.plot(x, snorm.pdf(x), ls=':', label='Skew norm')

    f_core = (snorm.ppf(.01), snorm.ppf(.99))
    # cc = CauchyConvolution(0., fwhm_p)
    gt_3 = GaussTrap_12_5(3)
    # plt.plot(x, cc.convolve(x, snorm.pdf, f_core, gt_3.quad),
    #     c='k', ls='--', lw=2, alpha=.6, label='CC 12_5')

    # Here sig is the std dev'n.
    sv = SkewVoigt(0., fwhm_p, sig, skew)
    plt.plot(x, sv.profile(x), c='b', ls='-', lw=1, alpha=.6, label='SkewVoigt')

    plt.legend()

    # How far off from actual skew norm mode?
    def obj(x):
        return -sv.snorm.pdf(x)
    mode = minimize(obj, 0.).x
    print('Actual snorm mode (frxn err):', mode, -mode/sv.loc_dop)
    scatter([mode], [sv.snorm.pdf([mode])])

    # Mode of the profile:
    def obj(x):
        return -sv.profile([x])
    mode = minimize(obj, 0.).x
    print('Actual SV mode (frxn err):', mode, -mode/sv.loc_dop)
    scatter([mode], [sv.profile([mode])])

