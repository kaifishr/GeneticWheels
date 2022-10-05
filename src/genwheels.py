import time
import math
import random

import numpy as np

from Box2D.examples.framework import (Framework, main)
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape)
# from simple_framework import SimpleFramework
from Box2D.Box2D import b2World, b2Vec2, b2Body, b2Filter

from torch.utils.tensorboard import SummaryWriter

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

    Creates convex 16-sided polygon in shape of square to body.
    Wheels are initially approximated as squares.
    
    """
    _vertices = [
        (-0.5, -0.5), (-0.5, -0.25),  (-0.5, 0.0),  (-0.5, 0.25),
        (-0.5, 0.5),  (-0.25, 0.5),  (0.0, 0.5),  (0.25, 0.5),
        (0.5, 0.5),  (0.5, 0.25),  (0.5, 0.0),  (0.5, -0.25),
        (0.5, -0.5),  (0.25, -0.5),  (0.0, -0.5),  (-0.25, -0.5)
    ]
    # d = 0.25
    # vertices = [
    #     (-0.5, -0.5), (-0.5 + d, -0.25),  (-0.5, 0.0),  (-0.5 + d, 0.25),
    #     (-0.5, 0.5),  (-0.25, 0.5 + d),  (0.0, 0.5),  (0.25, 0.5 + d),
    #     (0.5, 0.5),  (0.5 + d, 0.25),  (0.5, 0.0),  (0.5 + d, -0.25),
    #     (0.5, -0.5),  (0.25, -0.5 + d),  (0.0, -0.5),  (-0.25, -0.5 + d)
    # ]

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

        self.vertices = self.get_vertices()
        self.fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=self.vertices),
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1)
        )

        self.fixture = self.body.CreateFixture(self.fixture_def)

    def get_vertices(self) -> list:
        """Creates base vertices for wheel."""
        return [(self.diam * x, self.diam * y) for (x, y) in self._vertices]

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

    def mutate(self, vertices: list) -> None:
        """Mutates wheel's vertices."""
        self.body.DestroyFixture(self.fixture)
        eta = 1e-2
        self.vertices = [(eta * random.gauss(0, 1) + x, eta * random.gauss(0, 1) + y) for (x, y) in vertices]
        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=self.vertices), 
            density=1,
            filter=b2Filter(groupIndex=-1),
        )
        self.body.CreateFixture(fixture_def)

        for fixture in self.body.fixtures:
            print(sum([1 for _ in fixture.shape]))


class GeneticWheels(Framework):
# class GeneticWheels(SimpleFramework):

    name = "Genetic Wheels"
    description = "Rediscovering the wheel with genetic optimization."

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        n_wheels = config.optimizer.n_wheels
        self.n_max_iterations = config.optimizer.n_max_iterations

        self.world.gravity = (self.config.env.gravity.x, self.config.env.gravity.y)
        self.wheels = [Wheel(world=self.world, config=self.config) for _ in range(n_wheels)]
        self.inclined_plane = InclinedPlane(world=self.world, config=self.config)

        self.writer = SummaryWriter()
        self.iteration = 0
        self.generation = 0

    def reset(self) -> None:
        """Resets all wheels to initial parameter.
        """
        for wheel in self.wheels:
            wheel.reset()
    
    def comp_fitness(self) -> float:
        """Computes maximum fitness of wheels.

        The fitness is determined by the vertical
        distance traveled by the wheel.

        This method is called after wheels have stoped
        moving or if maximum number of iterations is 
        reached.
        
        Returns:
            List holding fitness scores.
        """
        scores = [wheel.body.position.x for wheel in self.wheels]
        idx_best = np.argmax(scores)
        return idx_best, scores[idx_best]

    def is_awake(self) -> bool:
        """Checks if wheels in simulation are awake.

        Returns: 
            True if at least one body is awake.
        """
        for wheel in self.wheels:
            if wheel.body.awake:
                return True

        return False

    def mutate(self, idx_best: int) -> None:
        """Mutates vertices of wheels."""
        # Get vertices of best wheel:
        vertices = self.wheels[idx_best].vertices
        # Pass best vertices to all wheels
        for wheel in self.wheels:
            wheel.mutate(vertices)

    def Step(self, settings):
        super(GeneticWheels, self).Step(settings)

        if not self.is_awake() or (self.iteration + 1) % self.n_max_iterations == 0:
            idx_best, max_score = self.comp_fitness()
            self.mutate(idx_best)
            self.writer.add_scalar("Score", max_score, self.generation)
            self.reset()
            self.iteration = 0
            self.generation += 1

        self.iteration += 1
