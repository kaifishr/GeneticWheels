DEBUG = False

import time
import random

import numpy as np

if DEBUG:
    from Box2D.examples.framework import Framework
else:
    from src.framework import SimpleFramework as Framework

from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.body import InclinedPlane, Wheel


class GeneticWheels(Framework):
    """Optimizer class for GeneticWheels"""

    name = "Genetic Wheels"
    description = "Rediscovering the wheel with genetic optimization."

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        n_wheels = config.optimizer.n_wheels
        self.n_max_iterations = config.optimizer.n_max_iterations

        self.world.gravity = (self.config.env.gravity.x, self.config.env.gravity.y)
        self.wheels = [
            Wheel(world=self.world, config=self.config) for _ in range(n_wheels)
        ]
        self.inclined_plane = InclinedPlane(world=self.world, config=self.config)

        self.writer = SummaryWriter()
        self.iteration = 0
        self.generation = 0

    def reset(self) -> None:
        """Resets all wheels to initial parameter."""
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
        return idx_best, scores[idx_best] - self.config.env.wheel.init_position.x

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

    def _step(self) -> None:
        """Performs single optimization step."""
        t_0 = time.time()

        if not self.is_awake() or (self.iteration + 1) % self.n_max_iterations == 0:
            idx_best, max_score = self.comp_fitness()
            self.mutate(idx_best)
            self.reset()

            self.writer.add_scalar("Score", max_score, self.generation)
            self.writer.add_scalar(
                "Time_Generation", time.time() - t_0, self.generation
            )

            self.iteration = 0
            self.generation += 1

        self.iteration += 1

    def Step(self, settings):
        super(GeneticWheels, self).Step(settings)
        self._step()

    def run(self) -> None:
        if DEBUG:
            super().run()
        else:
            while True:
                # Physics and rendering
                self.step()
                # Optimization
                self._step()
