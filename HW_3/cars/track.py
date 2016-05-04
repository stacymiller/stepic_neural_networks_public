from cmath import rect, phase
from math import ceil

import numpy as np
from numpy import pi

sectors = 48
radius = 5
width = 3
scale = radius / 5


# np.random.seed(5)

def get_partition(n, a, b=None):
    if b is None:
        b = a
        a = 0
    sample = np.random.rand(n)
    return a + (b - a) * np.cumsum(sample / sample.sum())


def generate_map(sectors, radius, width, scale):
    """
    :param sectors: number of sectors in the map
    :param radius: average distance between 0 and inner point of map
    :param width: distance between inner and outer points of map
    :param scale: scale of radius variation, as in np.random.normal(loc=radius, scale=scale, size=sectors)
    :return: list of tuples (`inner_point`, `outer_point`) of length :param sectors:
    """
    sector_angles = get_partition(sectors, -pi, pi)
    sector_radii = np.random.normal(loc=radius, scale=scale, size=sectors)
    sector_radii[sector_radii <= 0] = 1e-6
    inner_points = [rect(r, phi) for phi, r in zip(sector_angles, sector_radii)]
    outer_points = [rect(r, phi) for phi, r in zip(sector_angles, sector_radii + width)]
    return list(zip(inner_points, outer_points))


def plot_map(m, screen, scale=None, color=(0, 0, 0), width=2):
    if not scale:
        xmax, ymax = np.array([(abs(outer.real), abs(outer.imag)) for inner, outer in m]).max(axis=0)
        scale = ceil(xmax) + ceil(ymax) * 1j
    size = screen.get_width(), screen.get_height()
    from cars.utils import to_px
    points = np.array([[to_px(inner, scale, size), to_px(outer, scale, size)] for inner, outer in m])
    import pygame
    pygame.draw.polygon(screen, color, points[:, 0], width)
    pygame.draw.polygon(screen, color, points[:, 1], width)

    return scale
