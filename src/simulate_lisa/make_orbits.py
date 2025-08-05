"""
Filename: make_orbits.py
Author: William Mostrenko
Created: 2025-06-16
Description: Simulate LISA orbits and write to file.
"""

import os
from lisaorbits import KeplerianOrbits

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_orbit_data = os.path.join(PATH_bethLISA, "dist/orbit_data/")

orbits_fn = "orbits.h5"
lisa_orbits = KeplerianOrbits(t_init=0)

if os.path.exists(orbits_fn):
    os.remove(orbits_fn)
lisa_orbits.write(PATH_orbit_data + orbits_fn, mode='x')
