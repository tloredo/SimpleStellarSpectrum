"""
Experiment with the ssspec (Simple Stellar Spectrum) module

A simple model of a stellar spectrum with absorption lines, for testing
spectrum analysis algorithms.

Created 2024-04-30 by Tom Loredo
"""
import numpy as np
from numpy import *
import scipy
from scipy.special import wofz, voigt_profile

from astropy import constants as const

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

from ssspec import c, k_B, amu, ang2cm, amasses
from ssspec import BBSpecRad, T_quiet, T_spot, bb_sol, bb_spot
from ssspec import atmos, atmos_spot, atmos_fac
from ssspec import doppler_std, SpectralLine, SpectraSet
from ssspec import emergent_intens

from attrdict import ImmutableAttrDict


try:
    import myplot
    from myplot import ipy_ion, close_all, csavefig
    ipy_ion()
    #myplot.tex_on()
    csavefig.save = False
except ImportError:
    ion()


rt2 = sqrt(2)

# Full wavelength & frequency ranges of interest:
lam_ll, lam_uu = 2000, 7000
nu_ll, nu_uu = c/lam_uu/ang2cm, c/lam_ll/ang2cm

# Calcium line data:

# Wavelengths, frequencies:
lam_H, lam_K = 3968.47, 3933.66  # Wikipedia; vacuum wavelengths?
nu_H, nu_K = c/lam_H/ang2cm, c/lam_K/ang2cm

# Wavelength, freq. ranges spanning Ca H & K lines:
lam_l, lam_u = 3910., 4000.
nu_l, nu_u = c/lam_u/ang2cm, c/lam_l/ang2cm

# 1 K Doppler widths (std dev'n); width grows like sqrt(T):
am_Ca = amasses['Ca']
w_dop_1_H = sqrt(k_B/(am_Ca*amu*c**2)) * nu_H
w_dop_1_K = sqrt(k_B/(am_Ca*amu*c**2)) * nu_K

wdop_H = w_dop_1_H*sqrt(atmos.T)
wdop_K = w_dop_1_K*sqrt(atmos.T)

# TODO:  Switch to using doppler_std above.


# Gamma_damping = 1.e7  # Milic's value for Fe


# Plot black body spectra.

if False:
    # BB spectra, first vs. freq:
    bbfig = figure(figsize=(14,10))
    subplots_adjust(hspace=.3, wspace=.25)

    subplot(221)
    xlabel('Frequency (Hz)')
    ylabel('BB spectral radiance (per Hz)')
    nus = linspace(nu_ll, nu_uu, 1001)
    plot(nus, bb_sol.rad_nu(nus), label='Solar')
    plot(nus, bb_spot.rad_nu(nus), label='Spot')

    # Vs. wavelength, using BB scale units to do conversion:
    subplot(222)
    xlabel(r'$\lambda$ (ang)')
    ylabel(r'BB spectral radiance (per $\AA$)')
    lams = linspace(lam_ll, lam_uu, 1001)
    plot(lams, bb_sol.rad_l(lams), label='Solar')
    plot(lams, bb_spot.rad_l(lams), label='Spot')

    # BB(freq) converted to wavelength:
    subplot(223)
    xlabel(r'$\lambda$ (ang)')
    ylabel(r'BB spectral radiance (per $\AA$)')
    jac = c/(lams*ang2cm)**2  # for radiance per cm wavelength
    jac = ang2cm*jac  # convert to per angstrom
    plot(lams, jac*bb_sol.rad_nu(c/lams/ang2cm), label='Solar')
    plot(lams, jac*bb_spot.rad_nu(c/lams/ang2cm), label='Spot')



# Plot a hi-res solar spectrum from BASS2000 in the vicinity
# of Ca H & K.
bass_CaHK = loadtxt('bass2000_CaHK_3913_3993.txt')

fig = figure(figsize=(9,7))
xlabel(r'$\lambda$ ($\AA$)')
ylabel('Normalized intensity')
plot(bass_CaHK[:,0], bass_CaHK[:,1], ls='-')
title('BASS2000 Hi-Res Solar Spectrum')
xlim(lam_l, lam_u)  # to match the scale with plots below

# Define Ca H & K Fraunhofer lines.

def make_line(amass, ang, T, a, name=''):
    """
    Create a SpectralLine instance for a line from an element with atomic mass
    `amass`, with a line wavelength `ang` (angstroms), and with natural width
    determined by the Doppler width at temperature `T` (K) and the Voigt
    parameter `a` (roughly, the ratio of the natural width to the Doppler
    width, tyically << 1. for solar lines).
    """
    hz = c/(ang*ang2cm)
    fwhm = 2.*a*rt2*doppler_std(amass, T_quiet, hz)
    return SpectralLine(amass, ang=ang, fwhm_f=fwhm, name=name)

# Compute natural width from Doppler width at a nominal temperature,
# considering the Voigt `a` parameter to be specified at that temperature.
a = 1.e-0  # Linsky & Avrett report a ~ .001; using wider here
# fwhm = 2.*a*rt2*doppler_std(am_Ca, T_quiet, nu_H)
# Ca_H = SpectralLine(am_Ca, hz=nu_H, fwhm_f=fwhm, name='Ca H')
Ca_H = make_line(am_Ca, lam_H, T_quiet, a, 'Ca H')

# fwhm = 2.*a*rt2*doppler_std(am_Ca, T_quiet, nu_K)
# Ca_K = SpectralLine(am_Ca, hz=nu_K, wt=2., fwhm_f=fwhm, name='Ca K')
Ca_K = make_line(am_Ca, lam_K, T_quiet, a, 'Ca K')

# Experiment just with the deep, wide Ca lines.
lines = [Ca_H, Ca_K]

# Define the freq/wavelength grid by oversampling the line's Doppler width
# (which will be narrower than the post-atmosphere line).
# Note that a particular element with many lines will have ~const Doppler width
# across its lines; this isn't true for wavelength.
dop_samp_fac = 6
d_nu = doppler_std(am_Ca, T_quiet, nu_H)/dop_samp_fac
n_nu = int((nu_u - nu_l)/d_nu)
nus = linspace(nu_l, nu_u, n_nu)
lambdas = linspace(lam_l, lam_u, n_nu)

print('Doppler beta, Ca, p: %.3g  %.3g' % \
    (sqrt(k_B*T_quiet/(am_Ca*amu))/c, sqrt(k_B*T_quiet/amu)/c))
print('Grid:  [{:.7g}, {:.7g}] {}'.format(nu_l, nu_u, n_nu))
print('sig_dop/grid spacing:', doppler_std(am_Ca, T_quiet, nu_H)/(nus[1]-nus[0]))

fig = figure(figsize=(9,7))
xlabel('Frequency (Hz)')
ylabel('Profile')
title('Profile vs. frequency')

phi = zeros_like(nus)
for line in lines:
    print(line.name, line.hz, line.fwhm_f, line.wt)
    axvline(line.hz, ls=':')
    phi += line.profile_f(nus, T_quiet, pdf=False)

plot(nus, phi)


def mkinset(ax, box, xlims, ylims, ts=10, zoom=True):
    ax_in = ax.inset_axes(box)
    ax_in.set_xlim(*xlims)
    ax_in.set_ylim(*ylims)
    ax_in.tick_params(axis='both', which='major', labelsize=10)
    ax_in.xaxis.offsetText.set_fontsize(ts)
    ax_in.yaxis.offsetText.set_fontsize(ts)
    if zoom:
        ax.indicate_inset_zoom(ax_in)
    return ax_in


fig, ax = subplots(figsize=(9,7))
xlabel('$\\lambda$ (\u00c5)')
ylabel('Profile')
title('Profile vs. wavelength')

phi = zeros_like(lambdas)
for line in lines:
    print(line.name, line.ang, line.fwhm_w, line.wt)
    axvline(line.ang, ls=':')
    phi += line.profile_w(lambdas, T_quiet, pdf=False)

plot(lambdas, phi)

# ax_in = mkinset(ax, [0.36, 0.665, 0.275, 0.275], (3933.568, 3933.755),
#             (0., 5.7e9))
# ax_in.plot(lambdas, phi)

# ax_in = mkinset(ax, [0.65, 0.15, 0.275, 0.275], (3968.375, 3968.575),
#             (0., 2.9e9))
# ax_in.plot(lambdas, phi)



# Adjust line weights:
lines[0].wt = 10000.
lines[1].wt = 20000.
# lines[1].v_t = 100000000.*1e5

fig, ax = subplots(figsize=(9,7))
xlabel('$\\lambda$ (\u00c5)')
ylabel('Emergent intensity')
for line in lines:
    print(line.name, line.ang, line.fwhm_w, line.wt)
    axvline(line.ang, ls=':')

eintens = emergent_intens(lambdas, atmos, lines)
plot(lambdas, eintens, '-', label='Full')
cintens = emergent_intens(lambdas, atmos)
plot(lambdas, cintens, '-', label='Continuum')
legend()

# Normalize to the line-free continuum:

fig, ax = subplots(figsize=(9,7))
xlabel('$\\lambda$ (\u00c5)')
ylabel('Relative intensity')

plot(lambdas, eintens/cintens, '-')


# Continuum over large wavelength range:
fig, ax = subplots(figsize=(9,7))
xlabel('$\\lambda$ (\u00c5)')
ylabel('Continuum emergent intensity')

llambdas = linspace(1.e3, 2.e4, n_nu)
cintens = emergent_intens(llambdas, atmos)
plot(llambdas, cintens, '-', label='Continuum')
plot(llambdas, bb_sol.rad_w(llambdas), ':', label='%d K BB' % bb_sol.temp)
legend()

# Test how copy/deepcopy work with namedtuple instances with arrays:
if False:
    from copy import copy, deepcopy
    print('-------- Copy test -------')
    atmos1 = atmos
    print('T1: ', atmos1.T[:2])
    atmos2 = copy(atmos)
    atmos3 = deepcopy(atmos)

    atmos2.T[:] *= 2.
    print('\nCopy: ', atmos1.T[:2], '\n     ', atmos2.T[:2])

    atmos3.T[:] *= 2.
    print('\nDeep: ', atmos1.T[:2], '\n     ', atmos3.T[:2])


# Wider spectrum, to test overlap with Ca lines and isolated lines.
nu_l2 = nu_l - (nu_u - nu_l)  # double the Ca doublet span
n_nu = int((nu_u - nu_l)/d_nu)
nus = linspace(nu_l, nu_u, n_nu)
lam_u2 = lam_u + (lam_u - lam_l)
lambdas = linspace(lam_l, lam_u2, n_nu)

fig, ax = subplots(figsize=(9,7))
xlabel('$\\lambda$ (\u00c5)')
ylabel('Emergent intensity')


# Quiet and spot spectra with more lines:

# Depths of highly saturated Ca H & K:
Ca_H.wt = 1700.
Ca_K.wt = 3400.

# Made-up lines overlapping Ca H & K:

# H lines have wide Doppler widths.
H_a = make_line(amasses['H'], 3921., T_quiet, .1, 'H_a')
H_a.wt = .3
lines.append(H_a)
H_b = make_line(amasses['H'], 3934.9, T_quiet, .1, 'H_b')
H_b.wt = .7
lines.append(H_b)
H_c = make_line(amasses['H'], 3971., T_quiet, .03, 'H_c')
H_c.wt = .5
lines.append(H_c)

# O lines have intermediate Doppler widths.
O_a = make_line(amasses['O'], 3948., T_quiet, .1, 'O_a')
O_a.wt = .2
lines.append(O_a)
O_b = make_line(amasses['O'], 3964., T_quiet, .03, 'O_b')
O_b.wt = .4
lines.append(O_b)

# Hg lines have narrow Doppler widths.
Hg_a = make_line(amasses['Hg'], 3954.5, T_quiet, .1, 'Hg_a')
Hg_a.wt = .2
lines.append(Hg_a)
Hg_b = make_line(amasses['Hg'], 3966., T_quiet, .1, 'Hg_b')
Hg_b.wt = .4
lines.append(Hg_b)


# Made-up lines redward of Ca H & K:
H_d = make_line(amasses['H'], 3995.1, T_quiet, .1, 'H_d')
H_d.wt = .4
lines.append(H_d)
H_e = make_line(amasses['H'], 4070.7, T_quiet, .05, 'H_e')
H_e.wt = .2
lines.append(H_e)
H_f = make_line(amasses['H'], 4079.4, T_quiet, .07, 'H_f')
H_f.wt = .6
lines.append(H_f)

O_c = make_line(amasses['O'], 3995.3, T_quiet, .01, 'O_c')
O_c.wt = .3
lines.append(O_c)
O_d = make_line(amasses['O'], 4048.9, T_quiet, .03, 'O_d')
O_d.wt = .4
lines.append(O_d)

Hg_c = make_line(amasses['Hg'], 4020.3, T_quiet, .1, 'Hg_c')
Hg_c.wt = .2
lines.append(Hg_c)
Hg_d = make_line(amasses['Hg'], 4020.35, T_quiet, .01, 'Hg_d')
Hg_d.wt = .4
lines.append(Hg_d)
Hg_e = make_line(amasses['Hg'], 4033.8, T_quiet, .01, 'Hg_e')
Hg_e.wt = .6
lines.append(Hg_e)

# Widen the lines by adding a turbulence component; scale it to mass
# to keep diversity in the line widths.
wide = False
if wide:
    for line in lines:
        # ~150 km/s for H
        line.v_t = .0005*c/sqrt(line.amass)

eintens = emergent_intens(lambdas, atmos, lines)
cintens = emergent_intens(lambdas, atmos)

# Change some line weights and asymmetries for the spot.
H_b = 0.8
O_c.wt = .4
# O_c.asym = 1.  # asymmetric lines are currently VERY slow
O_d.wt = .6
# O_d.asym = -.6
eintens_s = emergent_intens(lambdas, atmos_spot, lines)
cintens_s = emergent_intens(lambdas, atmos_spot)

plot(lambdas, eintens, '-', label='Full')
plot(lambdas, cintens, '-', label='Continuum')
plot(lambdas, eintens_s, '-', label='Full (spot)')
plot(lambdas, cintens_s, '-', label='Cont. (spot)')
legend()

# Normalize to the line-free continuum:

fig, ax = subplots(figsize=(9,7))
xlabel('$\\lambda$ (\u00c5)')
ylabel('Relative intensity')

plot(lambdas, eintens/cintens, '-', label='Quiet', alpha=.7)
plot(lambdas, eintens_s/cintens_s, '-', label='Spot', alpha=.7)
legend()


if False:
    # A constant-T hot atmosphere, to produce wide lines:

    T = 5.e3*ones(2)
    logtau = array([-5., 1.37])
    hot_atmos = ImmutableAttrDict(T=T, log_tau_c=logtau)

    fig, ax = subplots(figsize=(9,7))
    xlabel('$\\lambda$ (\u00c5)')
    ylabel('Relative intensity')

    eintens_h = emergent_intens(lambdas, hot_atmos, lines)
    cintens_h = emergent_intens(lambdas, hot_atmos)

    plot(lambdas, eintens_h/cintens_h, '-', label='Full (hot)')
legend()


# Save a set of spectra.

# For faculae, use the spot line list; faculae tend to be associated
# with spots.
eintens_f = emergent_intens(lambdas, atmos_fac, lines)
cintens_f = emergent_intens(lambdas, atmos_fac)


sset = SpectraSet(lambdas=lambdas)
sset.full_q = eintens
sset.cont_q = cintens
sset.full_s = eintens_s
sset.cont_s = cintens_s
sset.full_f = eintens_f
sset.cont_f = cintens_f
if wide:
    sset.save('FAL-C-wide.h5')
else:
    sset.save('FAL-C-narrow.h5')