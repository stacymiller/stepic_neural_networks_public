import itertools
import random
from abc import ABCMeta, abstractmethod
from cmath import rect, pi, phase
from time import sleep

import numpy as np
import pygame

from cars.agent import SimpleCarAgent
from cars.track import plot_map
from cars.utils import CarState, to_px, rotate, intersect_ray_with_segment, draw_text, angle

black = (0, 0, 0)
white = (255, 255, 255)


class World(metaclass=ABCMeta):
    @abstractmethod
    def transition(self):
        pass

    @abstractmethod
    def run(self):
        pass


class SimpleCarWorld(World):
    COLLISION_PENALTY =  # выберите сами
    HEADING_REWARD =  # выберите сами
    WRONG_HEADING_PENALTY =  # выберите сами
    IDLENESS_PENALTY =  # выберите сами
    SPEEDING_PENALTY =  # выберите сами
    MIN_SPEED =  # выберите сами
    MAX_SPEED =  # выберите сами

    size = (800, 600)

    def __init__(self, num_agents, car_map, Physics, agent_class, **physics_pars):
        """
        Инициализирует мир
        :param num_agents: число агентов в мире
        :param car_map: карта, на которой всё происходит (см. track.py0
        :param Physics: класс физики, реализующий столкновения и перемещения
        :param agent_class: класс агентов в мире
        :param physics_pars: дополнительные параметры, передаваемые в конструктор класса физики
        (кроме car_map, являющейся обязательным параметром конструктора)
        """
        self.physics = Physics(car_map, **physics_pars)
        self.map = car_map

        # создаём агентов
        self.set_agents(num_agents, agent_class)

        self._info_surface = pygame.Surface(self.size)

    def set_agents(self, agents=1, agent_class=None):
        """
        Поместить в мир агентов
        :param agents: int или список Agent, если int -- то обязателен параметр agent_class, так как в мир присвоятся
         agents агентов класса agent_class; если список, то в мир попадут все агенты из списка
        :param agent_class: класс создаваемых агентов, если agents - это int
        """
        pos = (self.map[0][0] + self.map[0][1]) / 2
        vel = 0
        heading = rect(-0.3, 1)

        if type(agents) is int:
            self.agents = [agent_class() for _ in range(agents)]
        elif type(agents) is list:
            self.agents = agents
        else:
            raise ValueError("Parameter agent should be int or list of agents instead of %s" % type(agents))

        self.agent_states = {a: CarState(pos, vel, heading) for a in self.agents}
        self.circles = {a: 0 for a in self.agents}

        self._agent_surfaces = []
        self._agent_images = []

    def transition(self):
        """
        Логика основного цикла:
         подсчёт для каждого агента видения агентом мира,
         выбор действия агентом,
         смена состояния
         и обработка реакции мира на выбранное действие
        """
        for a in self.agents:
            vision = self.vision_for(a)
            action = a.choose_action(vision)
            next_agent_state, collision = self.physics.move(
                self.agent_states[a], action
            )
            self.circles[a] += angle(self.agent_states[a].position, next_agent_state.position) / (2*pi)
            self.agent_states[a] = next_agent_state
            a.receive_feedback(self.reward(next_agent_state, collision))

    def reward(self, state, collision):
        """
        Вычисление награды агента, находящегося в состоянии state.
        Эту функцию можно (и иногда нужно!) менять, чтобы обучить вашу сеть именно тем вещам, которые вы от неё хотите
        :param state: текущее состояние агента
        :param collision: произошло ли столкновение со стеной на прошлом шаге
        :return reward: награду агента (возможно, отрицательную)
        """
        a = np.sin(angle(-state.position, state.heading))
        heading_reward = 1 if a > 0.1 else a if a > 0 else 0
        heading_penalty = a if a <= 0 else 0
        idle_penalty = 0 if abs(state.velocity) > self.MIN_SPEED else -self.IDLENESS_PENALTY
        speeding_penalty = 0 if abs(state.velocity) < self.MAX_SPEED else -self.SPEEDING_PENALTY * abs(state.velocity)
        collision_penalty = - max(abs(state.velocity), 0.1) * int(collision) * self.COLLISION_PENALTY

        return heading_reward * self.HEADING_REWARD + heading_penalty * self.WRONG_HEADING_PENALTY + collision_penalty \
               + idle_penalty + speeding_penalty

    def eval_reward(self, state, collision):
        """
        Награда "по умолчанию", используется в режиме evaluate
        Удобно, чтобы не приходилось отменять свои изменения в функции reward для оценки результата
        """
        a = -np.sin(angle(-state.position, state.heading))
        heading_reward = 1 if a > 0.1 else a if a > 0 else 0
        heading_penalty = a if a <= 0 else 0
        idle_penalty = 0 if abs(state.velocity) > self.MIN_SPEED else -self.IDLENESS_PENALTY
        speeding_penalty = 0 if abs(state.velocity) < self.MAX_SPEED else -self.SPEEDING_PENALTY * abs(state.velocity)
        collision_penalty = - max(abs(state.velocity), 0.1) * int(collision) * self.COLLISION_PENALTY

        return heading_reward * self.HEADING_REWARD + heading_penalty * self.WRONG_HEADING_PENALTY + collision_penalty \
            + idle_penalty + speeding_penalty

    def run(self, steps=None):
        """
        Основной цикл мира; по завершении сохраняет текущие веса агента в файл network_config_agent_n_layers_....txt
        :param steps: количество шагов цикла; до внешней остановки, если None
        """
        scale = self._prepare_visualization()
        for _ in range(steps) if steps is not None else itertools.count():
            self.transition()
            self.visualize(scale)
            if self._update_display() == pygame.QUIT:
                break
            sleep(0.1)

        for i, agent in enumerate(self.agents):
            try:
                filename = "network_config_agent_%d_layers_%s.txt" % (i, "_".join(map(str, agent.neural_net.sizes)))
                agent.to_file(filename)
                print("Saved agent parameters to '%s'" % filename)
            except AttributeError:
                pass

    def evaluate_agent(self, agent, steps=1000, visual=True):
        """
        Прогонка цикла мира для конкретного агента (см. пример использования в комментариях после if _name__ == "__main__")
        :param agent: SimpleCarAgent
        :param steps: количество итераций цикла
        :param visual: рисовать картинку или нет
        :return: среднее значение награды агента за шаг
        """
        agent.evaluate_mode = True
        self.set_agents([agent])
        rewards = []
        if visual:
            scale = self._prepare_visualization()
        for _ in range(steps):
            vision = self.vision_for(agent)
            action = agent.choose_action(vision)
            next_agent_state, collision = self.physics.move(
                self.agent_states[agent], action
            )
            self.circles[agent] += angle(self.agent_states[agent].position, next_agent_state.position) / (2*pi)
            self.agent_states[agent] = next_agent_state
            rewards.append(self.eval_reward(next_agent_state, collision))
            agent.receive_feedback(rewards[-1])
            if visual:
                self.visualize(scale)
                if self._update_display() == pygame.QUIT:
                    break
                sleep(0.05)

        return np.mean(rewards)

    def vision_for(self, agent):
        """
        Строит видение мира для каждого агента
        :param agent: машинка, из которой мы смотрим
        :return: список из модуля скорости машинки, направленного угла между направлением машинки
        и направлением на центр и `agent.rays` до ближайших стен трека (запустите картинку, и станет совсем понятно)
        """
        state = self.agent_states[agent]
        vision = [abs(state.velocity), np.sin(angle(-state.position, state.heading))]
        extras = len(vision)

        delta = pi / (agent.rays - 1)
        start = rotate(state.heading, - pi / 2)

        sectors = len(self.map)
        for i in range(agent.rays):
            # define ray direction
            ray = rotate(start, i * delta)

            # define ray's intersections with walls
            vision.append(np.infty)
            for j in range(sectors):
                inner_wall = self.map[j - 1][0], self.map[j][0]
                outer_wall = self.map[j - 1][1], self.map[j][1]

                intersect = intersect_ray_with_segment((state.position, ray), inner_wall)
                intersect = abs(intersect - state.position) if intersect is not None else np.infty
                if intersect < vision[-1]:
                    vision[-1] = intersect

                intersect = intersect_ray_with_segment((state.position, ray), outer_wall)
                intersect = abs(intersect - state.position) if intersect is not None else np.infty
                if intersect < vision[-1]:
                    vision[-1] = intersect

            assert vision[-1] < np.infty, \
                "Something went wrong: {}, {}".format(str(state), str(agent.chosen_actions_history[-1]))
        assert len(vision) == agent.rays + extras, \
            "Something went wrong: {}, {}".format(str(state), str(agent.chosen_actions_history[-1]))
        return vision

    def visualize(self, scale):
        """
        Рисует картинку. Этот и все "приватные" (начинающиеся с _) методы необязательны для разбора.
        """
        for i, agent in enumerate(self.agents):
            state = self.agent_states[agent]
            surface = self._agent_surfaces[i]
            rays_lengths = self.vision_for(agent)[-agent.rays:]
            self._agent_images[i] = [self._draw_ladar(rays_lengths, state, scale),
                                     self._get_agent_image(surface, state, scale)]

        if len(self.agents) == 1:
            a = self.agents[0]
            draw_text("Reward: %.3f" % a.reward_history[-1], self._info_surface, scale, self.size,
                      text_color=white, bg_color=black)
            steer, acc = a.chosen_actions_history[-1]
            state = self.agent_states[a]
            draw_text("Action: steer.: %.2f, accel: %.2f" % (steer, acc), self._info_surface, scale,
                      self.size, text_color=white, bg_color=black, tlpoint=(self._info_surface.get_width() - 500, 10))
            draw_text("Inputs: |v|=%.2f, sin(angle): %.2f, circle: %.2f" % (
                abs(state.velocity), np.sin(angle(-state.position, state.heading)), self.circles[a]),
                      self._info_surface, scale,
                      self.size, text_color=white, bg_color=black, tlpoint=(self._info_surface.get_width() - 500, 50))

    def _get_agent_image(self, original, state, scale):
        angle = phase(state.heading) * 180 / pi
        rotated = pygame.transform.rotate(original, angle)
        rectangle = rotated.get_rect()
        rectangle.center = to_px(state.position, scale, self.size)
        return rotated, rectangle

    def _draw_ladar(self, sensors, state, scale):
        surface = pygame.display.get_surface().copy()
        surface.fill(white)
        surface.set_colorkey(white)
        start_pos = to_px(state.position, scale, surface.get_size())
        delta = pi / (len(sensors) - 1)
        ray = phase(state.heading) - pi / 2
        for s in sensors:
            end_pos = to_px(rect(s, ray) + state.position, scale, surface.get_size())
            pygame.draw.line(surface, (0, 255, 0), start_pos, end_pos, 2)
            ray += delta

        rectangle = surface.get_rect()
        rectangle.topleft = (0, 0)
        return surface, rectangle

    def _prepare_visualization(self):
        red = (254, 0, 0)
        pygame.init()
        screen = pygame.display.set_mode(self.size)
        screen.fill(white)
        scale = plot_map(self.map, screen)
        for state in self.agent_states.values():
            s = pygame.Surface((25, 15))
            s.set_colorkey(white)
            s.fill(white)
            pygame.draw.rect(s, red, pygame.Rect(0, 0, 15, 15))
            pygame.draw.polygon(s, red, [(15, 0), (25, 8), (15, 15)], 0)
            self._agent_surfaces.append(s)
            self._agent_images.append([self._get_agent_image(s, state, scale)])

        self._map_surface = screen
        return scale

    def _update_display(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return pygame.QUIT
        display = pygame.display.get_surface()
        display.fill(white)

        plot_map(self.map, display)
        for images in self._agent_images:
            for surf, rectangle in images:
                display.blit(surf, rectangle)
        display.blit(self._info_surface, (0, 0), None, pygame.BLEND_RGB_SUB)
        self._info_surface.fill(black)  # clear notifications from previous round
        pygame.display.update()


if __name__ == "__main__":
    from HW_3.cars.physics import SimplePhysics
    from HW_3.cars.track import generate_map

    np.random.seed(3)
    random.seed(3)
    m = generate_map(8, 5, 3, 3)
    SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2).run()

    # если вы хотите продолжить обучение уже существующей модели, вместо того,
    # чтобы создавать новый мир с новыми агентами, используйте код ниже:
    # # он загружает агента из файла
    # agent = SimpleCarAgent.from_file('filename.txt')
    # # создаёт мир
    # w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
    # # подключает к нему агента
    # w.set_agents([agent])
    # # и запускается
    # w.run()
    # # или оценивает агента в этом мире
    # print(w.evaluate_agent(agent, 500))
