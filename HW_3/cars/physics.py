from abc import ABCMeta, abstractmethod
from cmath import rect, pi

from cars.utils import CarState, get_line_coefs, to_line_equation, define_sector, Action, rotate


def to_polar(point):
    return point[0] + 1j * point[1]


class Physics(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def move(self, *args, **kwargs):
        """
        Moves the object from the state passed to function to the next state according to the rules of this physics.
        :param args: see implementation
        :param kwargs: see implementation
        :return: list of new coordinates for n ticks ahead (n is set on init)
        """
        pass


class SimplePhysics(Physics):
    def __init__(self, m, timedelta):
        """
        Creates instance of Physics with dummy rules
        :param m: m of car route in Euclidean coordinates; np.array with shape of (n_of_sectors, 2, 2)
        :param timedelta: discretization step
        :return:
        """
        self.map = m
        self.timedelta = timedelta

    def move(self, car_state, action, *args, **kwargs):
        """
        Moves object to the next point according to object's state. If object crosses the wall, move is rejected and
        object's position remains unchanged.
        :param car_state: state of car, of class CarState
        :param action: car action, of class Action
        :return: tuple(CarState with object's next position, boolean indication whether the collision happened)
        """
        position = car_state.position
        velocity = car_state.velocity
        acceleration = rotate(car_state.heading, action.steering * pi / 2) * action.acceleration
        new_position = position + velocity * self.timedelta + acceleration * (self.timedelta ** 2) / 2
        collision = self.is_out_of_map(new_position)
        if collision:
            return CarState(position, -0.5 * velocity, -car_state.heading), collision
        else:
            new_velocity = velocity + acceleration * self.timedelta
            heading = new_position - position
            if abs(heading) > 1e-5:
                heading /= abs(heading)
            else:
                heading = car_state.heading
            return CarState(new_position, new_velocity, heading), collision

    def collide(self, *args, **kwargs):
        pass

    def is_out_of_map(self, position):
        """
        Determine whether the point is inside the map or out of it
        :param position: the point in question, of class Complex
        :return: True if :param new_point: is out of map, else False
        """
        current_sector = define_sector(self.map, position)

        coefs = get_line_coefs(self.map[current_sector][0], self.map[current_sector - 1][0])
        sign_of_0 = to_line_equation(coefs, 0)
        sign_of_point = to_line_equation(coefs, position)
        if sign_of_0 * sign_of_point > 0:  # new point is on the same side of map's inner line as 0
            return True

        coefs = get_line_coefs(self.map[current_sector][1], self.map[current_sector - 1][1])
        sign_of_0 = to_line_equation(coefs, 0)
        sign_of_point = to_line_equation(coefs, position)
        if sign_of_0 * sign_of_point < 0:  # new point is on the other side of map's outer line than 0
            return True

        return False
