from src.bridge import CIRISWyomingHandler
from src.config import Config


def test_handler_init():
    config = Config()
    handler = CIRISWyomingHandler(config)
    assert handler.config == config
