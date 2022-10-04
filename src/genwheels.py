import time
import math
import random

import numpy as np

from Box2D.examples.framework import (Framework, main)
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape)
# from simple_framework import SimpleFramework
from Box2D.Box2D import b2World, b2Vec2, b2Body, b2Filter

from src.config import Config


class InclinedPlane:
    """Inclined plane class.

    Creates static inclined plane.
    """

    def __init__(self, world: b2World, config: Config):
        """Initializes the inclined plane."""
        cfg = config.env.inclined_plane
        x_0, y_0 = cfg.x_0, cfg.y_0
        x_1, y_1 = cfg.x_1, cfg.y_1

        world.CreateStaticBody(
            shapes=[
                b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)]),
            ]
        )


class Wheel:
    """Wheel class.

    Creates 16-sided polygon in shape of square to body.
    Wheels are initially approximated as squares.
    
    """
    vertices = [
        (-0.5, -0.5), (-0.5, -0.25),  (-0.5, 0.0),  (-0.5, 0.25),
        (-0.5, 0.5),  (-0.25, 0.5),  (0.0, 0.5),  (0.25, 0.5),
        (0.5, 0.5),  (0.5, 0.25),  (0.5, 0.0),  (0.5, -0.25),
        (0.5, -0.5),  (0.25, -0.5),  (0.0, -0.5),  (-0.25, -0.5)
    ]

    def __init__(self, world: b2World, config: Config):
        """Initializes the wheel class."""

        self.config = config
        self.diam = self.config.env.wheel.diam
        self.density = self.config.env.wheel.density
        self.friction = self.config.env.wheel.friction

        self.init_position = b2Vec2(
            config.env.wheel.init_position.x, 
            config.env.wheel.init_position.y
        )
        self.init_linear_velocity = b2Vec2(
            config.env.wheel.init_linear_velocity.x, 
            config.env.wheel.init_linear_velocity.y
        )
        self.init_angular_velocity = config.env.wheel.init_angular_velocity
        self.init_angle = (config.env.wheel.init_angle * math.pi) / 180.0

        self.body = world.CreateDynamicBody(
            bullet=False,
            allowSleep=True,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle = self.init_angle
        )

        self._add_wheel()

    def _add_wheel(self) -> None:
        """Adds 16-sided polygon in shape of square to body."""
        vertices = [(self.diam * x, self.diam * y) for (x, y) in self.vertices]
        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices), 
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1)
        )
        self.body.CreateFixture(fixture_def)

    def reset(self, noise: bool = False) -> None:
        """Resets wheel to initial position and velocity.
        """
        init_position = self.init_position
        init_linear_velocity = self.init_linear_velocity
        init_angular_velocity = self.init_angular_velocity
        init_angle = self.init_angle

        noise = self.config.env.wheel.noise

        if noise:
            # Position
            noise_x = random.gauss(mu=0.0, sigma=noise.position.x)
            noise_y = random.gauss(mu=0.0, sigma=noise.position.y)
            init_position += (noise_x, noise_y)

            # Linear velocity
            noise_x = random.gauss(mu=0.0, sigma=noise.linear_velocity.x)
            noise_y = random.gauss(mu=0.0, sigma=noise.linear_velocity.y)
            init_linear_velocity += (noise_x, noise_y)

            # Angular velocity
            noise_angular_velocity = random.gauss(mu=0.0, sigma=noise.angular_velocity)
            init_angular_velocity += noise_angular_velocity

            # Angle
            noise_angle = random.gauss(mu=0.0, sigma=noise.angle)
            init_angle += (noise_angle * math.pi) / 180.0

        self.body.position = init_position
        self.body.linearVelocity = init_linear_velocity
        self.body.angularVelocity = init_angular_velocity
        self.body.angle = init_angle


class GeneticWheel(Framework):
# class Bullet(SimpleFramework):

    name = "Genetic Wheels"
    description = "Rediscovering the wheel with genetic optimization."

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        n_wheels = config.optimizer.n_wheels

        self.world.gravity = (self.config.env.gravity.x, self.config.env.gravity.y)
        self.wheels = [Wheel(world=self.world, config=self.config) for _ in range(n_wheels)]
        self.inclined_plane = InclinedPlane(world=self.world, config=self.config)

        self.fitness = []

        self.iteration = 0

    def reset(self):
        for wheel in self.wheels:
            wheel.reset()
    # def _reset_objects(self):
    #     self._reset_bullet()
    #     self._reset_circles()

    # def _reset_bullet(self):
    #     self.bullet.DestroyFixture(self.bullet_fixture)
    #     self.bullet.CreateFixture(self.fixture_def)
    #     self.bullet.transform = [self.init_position, 0]
    #     self.bullet.linearVelocity = self.init_speed
    #     self.bullet.angularVelocity = 0.0

    # def _reset_circles(self):
    #     for circle, (x, y) in zip(self.circles, self.position):
    #         circle.transform = [(x, y), 0.0]
    #         circle.linearVelocity = (0.0, 0.0)
    #         circle.angularVelocity = 0.0

    # def _create_circle(self, pos):
    #     fixture = b2FixtureDef(shape=b2CircleShape(radius=self.circle_radius, pos=(0, 0)), 
    #                            density=self.circle_density, 
    #                            friction=self.circle_friction)

    #     self.circles.append(self.world.CreateDynamicBody(position=pos, fixtures=fixture))

    # def _generate_population(self):
    #     for _ in range(self.population_size):
    #         self.individuals.append(self.vertices)

    # def _mutate(self):
    #     buffer = list()
    #     for vertices in self.individuals:
    #         buffer.append([tuple((num + 10*np.random.uniform(-1, 1)) for num in item) for item in vertices])
    #     self.individuals = buffer

    # def _set_fixture_def(self):
    #     self.fixture_def = b2FixtureDef(shape=b2PolygonShape(vertices=self.vertices), density=self.density)

    def Step(self, settings):
        t0 = time.time()
        super(GeneticWheel, self).Step(settings)
        if self.iteration % 300 == 0:
            self.reset()
        print(time.time() - t0)
        self.iteration += 1
