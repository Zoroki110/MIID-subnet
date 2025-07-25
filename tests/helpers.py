# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Union, List, Dict
from bittensor import (
    Balance,
    NeuronInfo,
    AxonInfo,
    PrometheusInfo,
    __ss58_format__,
)
from bittensor.mock.wallet_mock import MockWallet as _MockWallet
from bittensor.mock.wallet_mock import get_mock_coldkey as _get_mock_coldkey
from bittensor.mock.wallet_mock import get_mock_hotkey as _get_mock_hotkey
from bittensor.mock.wallet_mock import get_mock_keypair as _get_mock_keypair
from bittensor.mock.wallet_mock import get_mock_wallet as _get_mock_wallet

try:
    from rich.console import Console
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover - stub for environments without rich
    class Console:
        """Minimal stub of rich.console.Console used for tests."""

        def __init__(self, *args, **kwargs):
            self._buffer = []

        def begin_capture(self):
            self._buffer = []

        def print(self, *args, **kwargs):
            self._buffer.append(" ".join(str(a) for a in args))

        def end_capture(self):
            output = "\n".join(self._buffer)
            self._buffer = []
            return output

    class Text:
        """Minimal stub of rich.text.Text used for tests."""

        def __init__(self, text: str = ""):
            self.plain = text

        @staticmethod
        def from_markup(text: str) -> "Text":
            return Text(text)

        @staticmethod
        def from_ansi(text: str) -> "Text":
            return Text(text)
import bittensor as bt
import torch
from MIID.protocol import IdentitySynapse


def __mock_wallet_factory__(*args, **kwargs) -> _MockWallet:
    """Returns a mock wallet object."""

    mock_wallet = _get_mock_wallet()

    return mock_wallet


class CLOSE_IN_VALUE:
    value: Union[float, int, Balance]
    tolerance: Union[float, int, Balance]

    def __init__(
        self,
        value: Union[float, int, Balance],
        tolerance: Union[float, int, Balance] = 0.0,
    ) -> None:
        self.value = value
        self.tolerance = tolerance

    def __eq__(self, __o: Union[float, int, Balance]) -> bool:
        # True if __o \in [value - tolerance, value + tolerance]
        # or if value \in [__o - tolerance, __o + tolerance]
        return (
            (self.value - self.tolerance) <= __o
            and __o <= (self.value + self.tolerance)
        ) or (
            (__o - self.tolerance) <= self.value
            and self.value <= (__o + self.tolerance)
        )


def get_mock_neuron(**kwargs) -> NeuronInfo:
    """
    Returns a mock neuron with the given kwargs overriding the default values.
    """

    mock_neuron_d = dict(
        {
            "netuid": -1,  # mock netuid
            "axon_info": AxonInfo(
                block=0,
                version=1,
                ip=0,
                port=0,
                ip_type=0,
                protocol=0,
                placeholder1=0,
                placeholder2=0,
            ),
            "prometheus_info": PrometheusInfo(
                block=0, version=1, ip=0, port=0, ip_type=0
            ),
            "validator_permit": True,
            "uid": 1,
            "hotkey": "some_hotkey",
            "coldkey": "some_coldkey",
            "active": 0,
            "last_update": 0,
            "stake": {"some_coldkey": 1e12},
            "total_stake": 1e12,
            "rank": 0.0,
            "trust": 0.0,
            "consensus": 0.0,
            "validator_trust": 0.0,
            "incentive": 0.0,
            "dividends": 0.0,
            "emission": 0.0,
            "bonds": [],
            "weights": [],
            "stake_dict": {},
            "pruning_score": 0.0,
            "is_null": False,
        }
    )

    mock_neuron_d.update(kwargs)  # update with kwargs

    if kwargs.get("stake") is None and kwargs.get("coldkey") is not None:
        mock_neuron_d["stake"] = {kwargs.get("coldkey"): 1e12}

    if kwargs.get("total_stake") is None:
        mock_neuron_d["total_stake"] = sum(mock_neuron_d["stake"].values())

    mock_neuron = NeuronInfo._neuron_dict_to_namespace(mock_neuron_d)

    return mock_neuron


def get_mock_neuron_by_uid(uid: int, **kwargs) -> NeuronInfo:
    return get_mock_neuron(
        uid=uid,
        hotkey=_get_mock_hotkey(uid),
        coldkey=_get_mock_coldkey(uid),
        **kwargs
    )


class MockStatus:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, *args, **kwargs):
        MockConsole().print(*args, **kwargs)


class MockConsole:
    """
    Mocks the console object for status and print.
    Captures the last print output as a string.
    """

    captured_print = None

    def status(self, *args, **kwargs):
        return MockStatus()

    def print(self, *args, **kwargs):
        console = Console(
            width=1000, no_color=True, markup=False
        )  # set width to 1000 to avoid truncation
        console.begin_capture()
        console.print(*args, **kwargs)
        self.captured_print = console.end_capture()

    def clear(self, *args, **kwargs):
        pass

    @staticmethod
    def remove_rich_syntax(text: str) -> str:
        """
        Removes rich syntax from the given text.
        Removes markup and ansi syntax.
        """
        output_no_syntax = Text.from_ansi(Text.from_markup(text).plain).plain

        return output_no_syntax


def get_mock_wallet():
    """Creates a mock wallet for testing"""
    wallet = bt.wallet(name="mock", hotkey="mock")
    return wallet


def create_mock_metagraph(n: int = 10):
    """
    Creates a mock metagraph with n nodes for testing
    
    Args:
        n (int): Number of nodes in the metagraph
    """
    metagraph = bt.metagraph(netuid=1)
    metagraph.hotkeys = [f"hotkey_{i}" for i in range(n)]
    metagraph.uids = torch.tensor(range(n))
    return metagraph


def create_test_synapse(
    names: List[str] = None,
    template: str = None
) -> IdentitySynapse:
    """
    Creates a test synapse with specified parameters
    
    Args:
        names (List[str]): List of names to generate variations for
        template (str): Query template to use
    """
    if names is None:
        names = ["John Smith"]
    if template is None:
        template = "Generate variations for {name}"
        
    return IdentitySynapse(
        names=names,
        query_template=template,
        variations={}
    )


def create_mock_responses(
    names: List[str],
    response_type: str = "valid"
) -> Dict[str, List[str]]:
    """
    Creates mock responses for testing
    
    Args:
        names (List[str]): List of names to generate responses for
        response_type (str): Type of response to generate (valid/empty/invalid)
    """
    if response_type == "valid":
        return {
            name: [f"{name}_var1", f"{name}_var2", f"{name}_var3"]
            for name in names
        }
    elif response_type == "empty":
        return {}
    elif response_type == "invalid":
        return {name: [] for name in names}
    else:
        raise ValueError(f"Unknown response type: {response_type}")


def mock_query_generator_config():
    """Creates a mock configuration for the QueryGenerator"""
    config = bt.config(withconfig=True)
    config.neuron.timeout = 120
    config.neuron.use_default_query = False
    return config
