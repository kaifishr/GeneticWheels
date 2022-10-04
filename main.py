from src.config import load_config
from src.genwheels import GeneticWheel


if __name__ == "__main__":
    config = load_config(path="config.yml")
    genetic_wheel = GeneticWheel(config=config)
    genetic_wheel.run()