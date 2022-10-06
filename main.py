from src.config import load_config
from src.genwheels import GeneticWheels
from src.utils import set_random_seed


if __name__ == "__main__":
    config = load_config(path="config.yml")
    set_random_seed(seed=config.seed)
    genetic_wheel = GeneticWheels(config=config)
    genetic_wheel.run()
