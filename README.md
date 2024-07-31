# SimpleStellarSpectrum
A simple model of a stellar spectrum with absorption lines, for testing spectrum analysis algorithms.

This collection of modules and scripts is organized into two parts (in corresponding directories):

* `StaticSpec` — Generate simplified *static* stellar spectra by solving the equation of radiative transfer for a simplified stellar atmosphere. The calculations use the temperature and optical depth profiles from the well-known **FAL C** solar model of Fontenla+ (1993).
* `DynSpecMix` — Generate a simplified time-dependent (dynamic) stellar spectrum via a time-dependent mixture of three static spectra computed using a script in `StaticSpec`, corresponding to simplified models of three components: the quiet Sun, a cooler sunspot region, and a slightly hotter facular region.

Users wanting to generate dynamic spectra need only use the `DynamicSpectraViaMixtures` IPython notebook in `DynSpecMix`, which includes a more detailed textual description of the calculations done in `StaticSpec`.


The main simplifying assumptions in the static spectrum calculations are:
* Plane-parallel geometry;
* Local thermodynamic equilibrium (LTE) line formation;
* Direct specification of line optical depths (i.e., ion densities are not explicitly calculated).

The example calculations provided here use Voigt profiles to compute the emergent line profiles. The `asym_voigt` module in `StaticSpec` includes capability to compute a new asymmetric generalization of the Voigt profile. Running that module as a script produces plots demonstrating this capability. Note that it is significantly more expensive to use asymmetric vs. symmetric profiles in the static spectrum calculation.

The user can freely specify a line list for the static spectrum calculations. The pre-calculated spectra provided in `DynSpecMix` include two highly saturated lines patterned after the Ca H and K Fraunhofer lines, and a set of shallower lines at arbitrary locations, and assigned atomic masses corresponding to elements producing other Fraunhofer lines.

This code was developed using a Conda/Mamba environment built as follows (`conda` may be used in place of `mamba`):

```
$ mamba create --name eprv10 -c conda-forge -c defaults python=3.10 ipython jupyter jupyterlab jupyterlab_widgets scipy matplotlib ipympl \
  h5py beautifulsoup4 html5lib bleach pandas sortedcontainers \
  pytz setuptools mpmath bottleneck jplephem asdf pyarrow colorcet hypothesis astropy pooch tqdm copier gsl
```

Activate it with `mamba activate eprv10`.

Our team uses this environment for general exoplanet EPRV experimentation, and not all of the packages listed above are required for the scripts and notebooks to run.



