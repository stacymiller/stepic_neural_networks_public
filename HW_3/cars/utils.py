from collections import namedtuple

# Create class point with cached properties polar and euclidian
# Point =
from cmath import rect, phase, pi
from math import atan2

import numpy as np
import pygame

CarState = namedtuple('CarState', ['position', 'velocity', 'heading'])
"""
:param position: car position on map, single Complex point
:param velocity: car speed, Complex
:param heading: where the car heading to (its view direction), Complex with length 1
"""

Action = namedtuple('Action', ['steering', 'acceleration'])
"""
:param steering: direction of wheel rotation
:param acceleration: acceleration =)
"""


def point(x, y):
    return x + y * 1j


def to_px(p, scale, size):
    """
    convert point from coordinate units to screen pixel units
    :param p: complex number representing point in coordinate units
    :param scale: complex number indicating how many coordinate units are from zero (center)
     to the right and to the top; zero is placed right in the center of the screen;
     ```scale=(1,1), size=(800,600)``` means that zero will be located at (400, 300) and
     ```p=-1j``` will be equal to pixel at (400, 600)
    :param size: int tuple indicating screen size ((800, 600), (1024,768), etc.)
    :return point: int tuple representing pixel ```p``` refers to
    """
    center = point(size[0] / 2, size[1] / 2)
    unit = int(center.real / scale.real) + int(center.imag / scale.imag) * 1j
    return int(center.real + unit.real * p.real), int(size[1] - center.imag - unit.imag * p.imag)


def from_px(x, y, scale, size):
    """
    convert point from screen pixel units to coordinate units
    :param x: x screen pixel units
    :param y: y screen pixel units
    :param scale: float (or int) tuple indicating how many coordinate units are from zero (center)
     to the right and to the top; zero is placed right in the center of the screen;
     ```scale=(1,1), size=(800,600)``` means that zero will be located at (400, 300) and
     ```x=400, y=600``` will be equal to point (0,-1)
    :param size: int tuple indicating screen size ((800, 600), (1024,768), etc.)
    :return point: Complex point in coordinate units
    """
    center = point(size[0] / 2, size[1] / 2)
    a = scale.real * (x - center.real) / center.real
    b = scale.imag * (size[1] - y - center.imag) / center.imag
    return point(a, b)


def rotate(p, phi):
    """rotates complex p in phi radians counter-clockwise (traditional way)"""
    return rect(abs(p), phase(p) + phi)


def get_line_coefs(p1, p2):
    """
    calculates coefficients of line that is determined by complex points p1 and p2;
    coefficients are normalized, i.e. A + B + C = 1
    :param p1: complex number
    :param p2: complex number
    :return: numpy array of shape (3,) with coefficients A, B, C such that
    A * p.real + B * p.imag + C = 0 for each p in [p1, p2]
    """
    assert p1 != p2, "Line cannot be determined by one point! p1 = {:.5f} = {:.5f} = p2".format(p1, p2)
    for _ in range(10):
        try:
            a = np.array([[p1.real, p1.imag, 1], [p2.real, p2.imag, 1], [1, 1, 1]])
            b = [0, 0, 1]
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            p1 += p1 - p2
    raise np.linalg.LinAlgError("Cannot count line coefs for line determined by {:.5f} and {:.5f}".format(p1, p2))


def to_line_equation(coefs, p):
    """
    Substitutes point p into line equation determined by coefs
    :param coefs: list or tuple of 3 elements (or anything that can be unzipped to 3 elements)
    :param p: point
    :return: A * p.real + B * p.imag + C
    """
    A, B, C = coefs
    return A * p.real + B * p.imag + C


def intersect(l1, l2):
    """
    Seeks for the point of intersection of two lines determined by their coefficients l1, l2
    :param l1: list or tuple of 3 elements [a1, b1, c1]
    :param l2: list or tuple of 3 elements [a2, b2, c2]
    :return: point of intersection of lines a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0
    """
    from numpy.linalg import solve
    a = [l1[:2], l2[:2]]
    b = [-l1[-1], -l2[-1]]
    ans = solve(a, b)
    return ans[0] + ans[1] * 1j


def intersect_ray_with_segment(ray, segment):
    """
    :param ray: (ray_start, ray_direction) -- tuple of complex numbers depicting ray that starts in point ray_start
    and proceeds in direction ray_direction
    :param segment: (segment_start, segment_end) -- tuple of complex numbers depicting segment that starts in
    segment_start and ends in segment_end
    :return: point of intersection of given line and segment, if any; None otherwise
    """
    r_start, r_dir = ray
    s_start, s_end = segment
    r_line = get_line_coefs(r_start, r_start + r_dir)
    s_line = get_line_coefs(s_start, s_end)
    intsct = intersect(r_line, s_line)
    if not (min(s_start.real, s_end.real) <= intsct.real <= max(s_start.real, s_end.real)):
        return None
    if not (min(s_start.imag, s_end.imag) <= intsct.imag <= max(s_start.imag, s_end.imag)):
        return None
    if r_dir.real * (intsct - r_start).real < 0 or r_dir.imag * (intsct - r_start).imag < 0:
        return None
    return intsct


def define_sector(m, position):
    cur_phase = phase(m[-1][0]) - 2 * pi
    for i in range(len(m)):
        prev_phase = cur_phase
        cur_phase = phase(m[i][0])
        if min(prev_phase, cur_phase) < phase(position) <= max(prev_phase, cur_phase):
            # position does not lie between i-1-th and i-th points of m
            return i
    raise AssertionError("phase(%s) = %f was not found anywhere in the m" % (str(position), phase(position)))


def draw_text(text, surface, scale, size, text_color=(0, 0, 0), bg_color=(255, 255, 255), tlpoint=None):
    """
    Draws text on surface
    :param text: string to draw
    :param surface: surface on which to draw
    :param size: int tuple indicating screen size ((800, 600), (1024,768), etc.)
    :param scale: float (or int) tuple indicating how many coordinate units are from zero (center)
     to the right and to the top (see `to_px`, `from_px`)
    :param text_color: text color
    :param bg_color: background color
    :param tlpoint: top left point of the text image; if None, 10px from lower right corner is selected
    """
    font = pygame.font.Font(None, 28)
    text_image = font.render(text, True, text_color, bg_color)
    text_width, text_height = font.size(text)
    if tlpoint is None:
        tlpoint = 10, size[1] - 10 - text_height
    elif type(tlpoint) is complex:
        tlpoint = to_px(tlpoint, scale, size)
    surface.blit(text_image, tlpoint)


def angle(x, y):
    """counts counter-clockwise angle between complex x and y"""
    # not using phase(y) - phase(x) based approach because of jump at -pi
    #  (consider phase(-0.999999 - 0.0000001j) and phase(-0.999999 + 0.0000001j))
    return atan2(x.real * y.imag - y.real * x.imag, x.real * y.real + x.imag * y.imag)
