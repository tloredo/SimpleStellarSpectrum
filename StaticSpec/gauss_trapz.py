"""
Hybrid Gauss-trapezoid quadrature.

Selected quadrature rules for computing definite integrals, from:

Hybrid Gauss-Trapezoidal Quadrature Rules | SIAM Journal on Scientific Computing
https://epubs.siam.org/doi/abs/10.1137/S1064827597325141
Bradley K. Alpert (1999)

These rules comprise a central (compound) trapezoid rule, and boundary rules
that replace the trapezoid rule with optimized nodes and weights, in the
manner of Gaussian quadrature.

Created 2024-03-29 by Tom Loredo
"""

import numpy as np
from numpy import *
import scipy


# Pre-computed boundary nodes, wts from Alpert's Tabel 6, O=12, a=5:


class GaussTrap_12_5:
    """
    Hybrid Gauss-trapezoid quadrature, order 12, offset parameter 5,
    with 6 nodes on each side of the interior trapezoid rule.

    This is an open rule; the end points are not among the nodes.
    """
 
    nodes_b = array([
        7.023955461621939e-02,
        4.312297857227970e-01,
        1.117752734518115e+00,
        2.017343724572518e+00,
        3.000837842847590e+00,
        4.000000000000000e+00 ])

    wts_b = array([
        1.922315977843698e-01,
        5.348399530514687e-01,
        8.170209442488760e-01,
        9.592111521445966e-01,
        9.967143408044999e-01,
        9.999820119661890e-01 ])

    n_b = 6
    a = 5.  # offset parameter for inner trapezoid rule

    def __init__(self, n_in):
        self.n_in = n_in
        self.n = 2*self.n_b + self.n_in
        self.h = 1./(n_in + 2*self.a - 1.)

        # lower, upper, inner nodes; then collected:
        nodes_l = self.nodes_b*self.h
        nodes_u = flip(1. - nodes_l)
        nodes_in = self.a*self.h + arange(n_in)*self.h
        self.nodes = concatenate((nodes_l, nodes_in, nodes_u))

        # Collect weights, using equal (trapezoid) inner weights.
        wts_in = ones(n_in)
        self.wts = self.h * concatenate((self.wts_b, wts_in, flip(self.wts_b)))

    def quad_unit(self, f, *args):
        """
        Quadrature of the function `f` over the unit interval, with respect
        to its first argument.  `args` may contain optional extra arguments.
        """
        return sum(self.wts*f(self.nodes, *args))

    def quad(self, l, u, f, xtra=False, *args):
        """
        Quadrature of the function `f` over [`l`, `u`].
        """
        w = u - l
        if xtra:
            nodes = l + w*self.nodes
            wts = w*self.wts
            fvals = f(nodes, *args)
            return sum(self.wts*fvals), nodes, wts, fvals
        else:
            return w*sum(self.wts*f(l + w*self.nodes, *args))


if __name__ == '__main__':
    
    import matplotlib as mpl
    from matplotlib.pyplot import *
    from scipy.stats import norm

    try:
        import myplot
        from myplot import ipy_ion, close_all, csavefig
        ipy_ion()
        #myplot.tex_on()
        csavefig.save = False
    except ImportError:
        ion()

    def f(x):
        """
        Integrates to 2 over [-1,1]; to interval size over any
        other interval.
        """
        return 1

    def g(x):
        """
        Integrates to 2 over [-1,1]; to u**3 - l**3 over [l,u].
        """
        return 3*x**2

    def h(x):
        """
        Integrates to 4 over [-1,1] (odd term vanishes); to  101000000887.6
        over [2, 10]; to 1+.1+1 = 2.1 over [0,1].
        """
        return 11*x**10 + x**9 + 3*x**2


    gt_3 = GaussTrap_12_5(3)
    print('Using {} nodes....'.format(gt_3.n))

    print('Polynomial cases:')
    print(gt_3.quad_unit(f), 1.)
    print(gt_3.quad_unit(g), 1.)
    print(gt_3.quad_unit(h), 2.1)

    print(gt_3.quad(2., 4., f), 2.)
    print(gt_3.quad(-1., 1., g), 2.)
    print(gt_3.quad(-1., 1., h), 4.)
    q = gt_3.quad(2., 10., h)
    q_t = 101000000887.6
    print(q, q_t, (q-q_t)/q_t)
    print()

    print('Normal PDF cases:')
    sn = norm()
    q = gt_3.quad(-1., 1., sn.pdf)
    q_t = sn.cdf(1.) - sn.cdf(-1.)
    print('1-sig:', q, q_t, (q-q_t)/q_t)
    q = gt_3.quad(-2., 2., sn.pdf)
    q_t = sn.cdf(2.) - sn.cdf(-2.)
    print('2-sig:', q, q_t, (q-q_t)/q_t)
    q = gt_3.quad(-3., 3., sn.pdf)
    q_t = sn.cdf(3.) - sn.cdf(-3.)
    print('3-sig:', q, q_t, (q-q_t)/q_t)
    q = gt_3.quad(-5., 5., sn.pdf)
    q_t = sn.cdf(5.) - sn.cdf(-5.)
    print('5-sig:', q, q_t, (q-q_t)/q_t)

    q = gt_3.quad(-1.5, 3., sn.pdf)
    q_t = sn.cdf(3.) - sn.cdf(-1.5)
    print('[-1.5, 3]:', q, q_t, (q-q_t)/q_t)
