# pyGenISW

Author:         Krishna Naidoo                          
Version:        1.0.0                               
Homepage:       https://github.com/knaidoo29/pyGenISW    

Computes the Integrated Sachs-Wolfe using spherical Bessel transforms for data
provided in healpix redshift slices.

## Dependencies

* numpy
* scipy
* healpy
* [TheoryCL](https://github.com/knaidoo29/TheoryCL)

## Installation

Clone the github repository:

```
git clone https://github.com/knaidoo29/pyGenISW
```

To install using pip:

```
cd pyGenISW
pip install -e . [--user]
```

or use setup:

```
cd pyGenISW
python setup.py build
python setup.py install
```

## Tutorial

```
import numpy as np
import healpy as hp
import pyGenISW

# setup basic cosmology using the TheoryCL package and creating look up tables
# for the linear growth functions

omega_m             = 0.25              # matter density
omega_l             = 1 - omega_m       # lambda density
h0                  = 0.7               # Hubble constant (H0/100)
zmin_lookup         = 0.                # minimum redshift for lookup table
zmax_lookup         = 10.               # maximum redshift for lookup table
zbin_num            = 10000             # number of points in the lookup table
zbin_mode           = 'log'             # either 'linear' or 'log' binnings in z (log means log of 1+z)

GISW = pyGenISW.isw.SphericalBesselISW()
GISW.cosmo(omega_m=omega_m, omega_l=omega_l, h0=h0)
GISW.calc_table(zmin=zmin_lookup, zmax=zmax_lookup, zbin_num=zbin_num, zbin_mode=zbin_mode)

# spherical Bessel Transform (SBT) setup

zmin                = 0.                # minimum redshift for the SBT
zmax                = 2.                # maximum redshift for the SBT
zedge_min           =                   # the minimum of each redshift slice
zedge_max           =                   # the maximum of each redshift slice
Lbox                = 3072.             # size of the simulation box
kmin                = 2.*np.pi/Lbox     # minimum k
kmax                = 0.1               # maximum k
lmax                = None              # if you want to specify the maximum l for the SBT transform
nmax                = None              # if you want to specify the maximum n for the SBT transform
uselightcone        = True              # set to True if the
boundary_conditions = 'normal'          # boundary conditions for the SBT, either 'normal' or 'derivative'

GISW.setup(zmin, zmax, zedge_min, zedge_max, kmin=kmin, kmax=kmax,
           lmax=lmax, nmax=nmax, uselightcone=uselightcone,
           boundary_conditions=boundary_conditions)

# converts each redshift slice into spherical harmonic alms
for i in range(0, len(zedge_min)):
   map_slice = # map of the density for redshift slice i corresponding to zedges_min[i]
   GISW.slice2alm(map_slice, i)

# converts spherical harmonic alms for the slices to the SBT coefficients
GISW.alm2sbt()

# you can store the SBT coefficients to avoid recomputing this again by running:
sbt_fname_prefix = # name of prefix for SBT file
GISW.sbt_save(prefix=sbt_fname_prefix)

# create ISW for contributions between zmin_isw and zmax_isw
alm_isw = GISW.sbt2isw_alm(zmin=zmin_isw, zmax=zmax_isw)

# convert alm to healpix map
nside = 256
map_isw = hp.alm2map(alm_isw, nside)*GISW.Tcmb
```

## Citing

If you use this package please provide a link to the github repository and cite
the following paper *ArXiv Link*.

## Support

If you have any issues with the code or want to suggest ways to improve it please open a new issue ([here](https://github.com/knaidoo29/pyGenISW/issues))
or (if you don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.
