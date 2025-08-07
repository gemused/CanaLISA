"""
Filename: make_orbits.py
Author: William Mostrenko
Created: 2025-06-16
Description: Simulate LISA orbits and write to file.
"""

import os
from lisaorbits import KeplerianOrbits, StaticConstellation

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_orbit_data = os.path.join(PATH_bethLISA, "dist/lisa_data/orbits_data/")

orbits_path = PATH_orbit_data + "orbits.h5"
orbits_position = KeplerianOrbits(t_init=10368000).compute_position([0], [1, 2, 3])[0]

lisa_orbits = StaticConstellation(orbits_position[0], orbits_position[1], orbits_position[2])

if os.path.exists(orbits_path):
    os.remove(orbits_path)
lisa_orbits.write(orbits_path, mode='x')
