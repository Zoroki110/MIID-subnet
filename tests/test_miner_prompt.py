import pytest

from neurons.miner import Miner


def create_dummy_miner():
    config = Miner.config()
    config.mock = True
    return Miner(config=config)


def test_build_prompt_contains_name_and_newline():
    miner = create_dummy_miner()
    prompt = miner.build_prompt(["alice"])
    assert "alice:" in prompt
    assert "\n" in prompt
