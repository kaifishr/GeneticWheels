import math
import random

from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape
from Box2D.Box2D import b2World, b2Vec2, b2Filter

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
                b2EdgeShape(vertices=[(x_1, y_1), (x_1, y_1 + 4.0)]),
                b2EdgeShape(vertices=[(x_0, y_0), (x_0, y_0 + 4.0)]),
            ]
        )


class Wheel:
    """Wheel class.

    Creates convex 16-sided polygon in shape of square to body.
    Wheels are initially approximated as squares.
    """

    _vertices = [
        (-0.5, -0.5),
        (-0.5, -0.25),
        (-0.5, 0.0),
        (-0.5, 0.25),
        (-0.5, 0.5),
        (-0.25, 0.5),
        (0.0, 0.5),
        (0.25, 0.5),
        (0.5, 0.5),
        (0.5, 0.25),
        (0.5, 0.0),
        (0.5, -0.25),
        (0.5, -0.5),
        (0.25, -0.5),
        (0.0, -0.5),
        (-0.25, -0.5),
    ]

    def __init__(self, world: b2World, config: Config):
        """Initializes the wheel class."""

        self.config = config
        self.diam = self.config.env.wheel.diam
        self.density = self.config.env.wheel.density
        self.friction = self.config.env.wheel.friction

        self.init_position = b2Vec2(
            config.env.wheel.init_position.x, config.env.wheel.init_position.y
        )
        self.init_linear_velocity = b2Vec2(
            config.env.wheel.init_linear_velocity.x,
            config.env.wheel.init_linear_velocity.y,
        )
        self.init_angular_velocity = config.env.wheel.init_angular_velocity
        self.init_angle = (config.env.wheel.init_angle * math.pi) / 180.0

        self.body = world.CreateDynamicBody(
            bullet=False,
            allowSleep=True,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle=self.init_angle,
        )

        self.vertices = [(self.diam * x, self.diam * y) for (x, y) in self._vertices]
        self.fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=self.vertices),
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1),
        )

        self.fixture = self.body.CreateFixture(self.fixture_def)

    def reset(self) -> None:
        """Resets wheel to initial position and velocity."""
        init_position = self.init_position
        init_linear_velocity = self.init_linear_velocity
        init_angular_velocity = self.init_angular_velocity
        init_angle = self.init_angle

        self.body.position = init_position
        self.body.linearVelocity = init_linear_velocity
        self.body.angularVelocity = init_angular_velocity
        self.body.angle = init_angle

    def mutate(self, vertices: list) -> None:
        """Mutates wheel's vertices."""
        self.body.DestroyFixture(self.fixture)

        p = self.config.optimizer.mutation_probability
        rho = self.config.optimizer.mutation_rate

        def _mutate(x: float) -> float:
            return x + (random.random() < p) * random.gauss(0, rho)

        def _clip(x: float) -> float:
            return min(x, 0.5 * self.diam)

        # Mutate vertices
        self.vertices = [(_mutate(x), _mutate(y)) for (x, y) in vertices]

        # Keep wheels from getting too big
        self.vertices = [(_clip(x), _clip(y)) for (x, y) in self.vertices]

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=self.vertices),
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1),
        )
        self.fixture = self.body.CreateFixture(fixture_def)
