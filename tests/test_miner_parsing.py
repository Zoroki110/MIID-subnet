import pytest

from neurons.miner import Miner


def create_dummy_miner():
    config = Miner.config()
    config.mock = True
    return Miner(config=config)


@pytest.mark.parametrize(
    "response,expected",
    [
        ("alice:a1,a2\nbob:b1,b2", {"alice": ["a1", "a2"], "bob": ["b1", "b2"]}),
        ("alice:a1,a2; bob:b1,b2", {"alice": ["a1", "a2"], "bob": ["b1", "b2"]}),
        ("1) alice - a1, a2; 2) bob - b1, b2", {"alice": ["a1", "a2"], "bob": ["b1", "b2"]}),
    ],
)
def test_process_batch_response(response, expected):
    miner = create_dummy_miner()
    result = miner.process_batch_response(response, list(expected.keys()), 0, "")
    for name, vars in expected.items():
        assert result[name][: len(vars)] == vars

