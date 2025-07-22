"""Minimal stub of the bittensor package for offline unit tests."""

from __future__ import annotations

from types import SimpleNamespace
import logging as _logging
import copy

# expose a standard logger similar to the real library
logging = _logging.getLogger("bittensor")

__ss58_format__ = 42


class Balance(float):
    pass


class TerminalInfo(SimpleNamespace):
    status_code: int = 0
    status_message: str = ""
    process_time: float = 0.0


class AxonInfo(SimpleNamespace):
    pass


class PrometheusInfo(SimpleNamespace):
    pass


class NeuronInfo(SimpleNamespace):
    @staticmethod
    def _neuron_dict_to_namespace(d: dict) -> "NeuronInfo":
        return NeuronInfo(**d)


class Synapse(SimpleNamespace):
    def copy(self) -> "Synapse":
        return copy.deepcopy(self)


class wallet:
    def __init__(self, name: str = "mock", hotkey: str = "mock", config=None):
        self.name = name
        self.hotkey = SimpleNamespace(ss58_address=f"{hotkey}_hotkey")
        self.coldkey = SimpleNamespace(ss58_address=f"{name}_coldkey")

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"wallet({self.hotkey.ss58_address})"


class MockWallet(wallet):
    pass


class axon(SimpleNamespace):
    pass


class dendrite:
    def __init__(self, wallet: wallet | None = None):
        self.wallet = wallet
        self.keypair = getattr(wallet, "hotkey", SimpleNamespace(ss58_address="0"))

    def preprocess_synapse_for_request(self, axon, synapse: Synapse, timeout: float):
        if not hasattr(synapse, "dendrite"):
            synapse.dendrite = TerminalInfo()
        return synapse


class subtensor:
    def __init__(self, network: str = "mock", config=None):
        self.network = network
        self.chain_endpoint = "mock_endpoint"
        self._subnets: dict[int, list[NeuronInfo]] = {}

    # subnet management helpers
    def create_subnet(self, netuid: int):
        self._subnets.setdefault(netuid, [])

    def subnet_exists(self, netuid: int) -> bool:
        return netuid in self._subnets

    def force_register_neuron(self, netuid: int, hotkey: str, coldkey: str, balance: int = 0, stake: int = 0):
        neuron = NeuronInfo(
            uid=len(self._subnets.get(netuid, [])),
            hotkey=hotkey,
            coldkey=coldkey,
            stake={coldkey: stake},
            total_stake=stake,
        )
        self._subnets.setdefault(netuid, []).append(neuron)
        return neuron

    def neurons(self, netuid: int):
        return list(self._subnets.get(netuid, []))

    def is_hotkey_registered(self, netuid: int, hotkey_ss58: str) -> bool:
        return any(n.hotkey == hotkey_ss58 for n in self._subnets.get(netuid, []))

    def metagraph(self, netuid: int):
        return metagraph(netuid=netuid, network=self.network, subtensor=self)

    def set_weights(self, *args, **kwargs):  # pragma: no cover - simple stub
        return True, "ok"

    def get_current_block(self):  # pragma: no cover - simple stub
        return 0


class MockSubtensor(subtensor):
    pass


class metagraph:
    def __init__(self, netuid: int = 1, network: str = "mock", sync: bool = True, subtensor: subtensor | None = None):
        self.netuid = netuid
        self.network = network
        self.subtensor = subtensor or subtensor(network=network)
        self.n = 0
        self.uids = []
        self.axons: list[AxonInfo] = []
        self.hotkeys: list[str] = []
        self.validator_permit = []
        self.S = []
        if sync:
            self.sync()

    def sync(self, subtensor: subtensor | None = None):
        if subtensor is not None:
            self.subtensor = subtensor
        neurons = self.subtensor.neurons(self.netuid)
        self.n = len(neurons)
        self.uids = list(range(self.n))
        self.hotkeys = [n.hotkey for n in neurons]
        self.axons = [AxonInfo(ip=0, port=0, is_serving=True) for _ in neurons]
        self.validator_permit = [True] * self.n
        self.S = [n.total_stake if hasattr(n, "total_stake") else 0 for n in neurons]


class Config(SimpleNamespace):
    def merge(self, other: "Config"):
        for k, v in other.__dict__.items():
            setattr(self, k, v)


class config(Config):
    def __init__(self, *args, **kwargs):
        kwargs.pop("withconfig", None)
        super().__init__(**kwargs)


def check_config(cfg):  # pragma: no cover - stub
    pass


# expose mock.wallet utilities
from .mock.wallet_mock import (
    MockWallet as MockWallet,
    get_mock_wallet,
    get_mock_hotkey,
    get_mock_coldkey,
    get_mock_keypair,
)

__all__ = [
    "logging",
    "__ss58_format__",
    "Balance",
    "TerminalInfo",
    "AxonInfo",
    "PrometheusInfo",
    "NeuronInfo",
    "Synapse",
    "wallet",
    "MockWallet",
    "axon",
    "dendrite",
    "subtensor",
    "MockSubtensor",
    "metagraph",
    "config",
    "Config",
    "check_config",
]
