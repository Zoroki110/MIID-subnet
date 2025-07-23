import bittensor as bt
from neurons.miner import Miner


def test_process_batch_response_with_spaces_and_newlines():
    miner = Miner(config=bt.config())
    response = "john  :  j1, j2 ;\n jane: j3 ,j4 ; bob :b1,b2"
    result = miner.process_batch_response(response, ["john", "jane", "bob"], 0, "")
    assert result["john"][:2] == ["j1", "j2"]
    assert result["jane"][:2] == ["j3", "j4"]
    assert result["bob"][:2] == ["b1", "b2"]
