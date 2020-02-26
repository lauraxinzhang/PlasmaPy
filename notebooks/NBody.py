#!/usr/bin/env python
# coding: utf-8

# In[1]:


from plasmapy import simulation
import astropy.units as u
import numpy as np


# In[2]:


import xarray


# In[3]:


from plasmapy.classes.sources.interparticle import InterParticleForces


# In[4]:


eq_distance = 0.1
forces = {key: InterParticleForces(key, 100, eq_distance, 1, 1e3, 0.01, 7) for key in ('python', 'njit', 'njit_parallel')}

from collections import namedtuple
CustomParticle = namedtuple('custom_particle', ['mass', 'charge'])
particle = CustomParticle(mass=1 * u.dimensionless_unscaled, charge=1 * u.dimensionless_unscaled)

L = 1 * u.m
N = 128
np.random.seed(0)
x = u.Quantity(np.random.random((N, 3))*L,  u.m)
v = u.Quantity(np.zeros(x.shape, dtype=float), u.m / u.s)

from scipy import spatial
tree = spatial.cKDTree(x)
close_pairs = tree.query_pairs(eq_distance)
while close_pairs:
    for a, b in close_pairs:
        x[b] = np.random.random(3) * L
    tree = spatial.cKDTree(x)
    close_pairs = tree.query_pairs(eq_distance)


# In[ ]:


solutions = {engine: simulation.ParticleTracker(forces[engine], x, v, particle).run(1e-3 * u.s, dt = 1e-6 * u.s) for engine in ['njit', 'njit_parallel']}
