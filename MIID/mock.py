import asyncio
import os
import random
import time
from typing import List

import bittensor as bt
from substrateinterface import Keypair

# ------------------------------------------------------------------
# Ensure bt.logging behaves like the builtin logging module for tests that
# use ``assertLogs(bt.logging, ...)``.  We inject minimal stubs when missing.
# ------------------------------------------------------------------

import logging as _logging

if not hasattr(bt.logging, "handlers"):
    bt.logging.handlers = []  # type: ignore[attr-defined]

if not hasattr(bt.logging, "level"):
    bt.logging.level = _logging.INFO  # type: ignore[attr-defined]

if not hasattr(bt.logging, "setLevel"):
    def _set_level(level):  # type: ignore[override]
        bt.logging.level = level  # type: ignore[attr-defined]
    bt.logging.setLevel = _set_level  # type: ignore[attr-defined]

# Needed so ``assertLogs`` can temporarily disable propagation
if not hasattr(bt.logging, "propagate"):
    bt.logging.propagate = False  # type: ignore[attr-defined]


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)

        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0.
        # pytest may construct several MockSubtensor instances during a run and
        # the underlying bittensor mock keeps a *global* chain_state.  If the
        # same hotkey is registered twice we get a "Hotkey already registered"
        # exception.  We simply ignore that situation.

        if wallet is not None:
            try:
                self.force_register_neuron(
                    netuid=netuid,
                    hotkey=wallet.hotkey.ss58_address,
                    coldkey=wallet.coldkey.ss58_address,
                    balance=100000,
                    stake=100000,
                )
            except Exception as e:
                if "already registered" not in str(e).lower():
                    raise

        # Register *exactly* n miner neurons. We first check how many miners
        # already exist (excluding the optional validator at uid 0).

        existing_neurons = self.neurons(netuid=netuid)
        existing_miner_count = len(existing_neurons) - (
            1 if wallet is not None else 0
        )

        miners_to_add = max(0, n - existing_miner_count)

        import uuid

        for _ in range(miners_to_add):
            unique_hotkey = f"miner-hotkey-{uuid.uuid4().hex[:6]}"
            try:
                self.force_register_neuron(
                    netuid=netuid,
                    hotkey=unique_hotkey,
                    coldkey="mock-coldkey",
                    balance=100000,
                    stake=100000,
                )
            except Exception as e:
                if "already registered" not in str(e).lower():
                    raise
        # ------------------------------------------------------------------
        # Ensure exactly *n* miners (+ optional validator) exist for this
        # subnet across repeated instantiations in the same test session.
        # ------------------------------------------------------------------

        desired_total = n + (1 if wallet is not None else 0)
        self._desired_total = desired_total  # Persist for neurons()/metagraph

    # ------------------------------------------------------------------
    # Override ``neurons`` so that tests see exactly the expected number of
    # entries even if global chain-state contains extras from previous runs.
    # ------------------------------------------------------------------

    def neurons(self, netuid: int):  # type: ignore[override]
        all_neurons = super().neurons(netuid)
        desired = getattr(self, "_desired_total", None)
        if desired is None:
            return all_neurons
        return all_neurons[:desired]


class MockMetagraph(bt.metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        # Public defaults used by test-suite assertions
        self.default_ip = "127.0.0.0"
        self.default_port = 8091

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        # Trim axons list to match requested subnet size when MockSubtensor has
        # been capped via _desired_total.
        desired = getattr(self.subtensor, "_desired_total", None)
        if desired is not None:
            self.axons = self.axons[:desired]

        for axon in self.axons:
            axon.ip = self.default_ip
            axon.port = self.default_port

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")


class MockDendrite(bt.dendrite):
    __slots__ = ("_session",)
    """
    A filesystem-free stand-in for ``bittensor.dendrite``.

    • If no wallet is supplied we generate a throw-away *Keypair* in memory so
      bittensor never tries to read ``~/.bittensor/...`` keyfiles.
    • We also create ``self._session = None`` so the upstream destructor
      doesn’t raise warnings.
    """

    def __init__(self, wallet=None):
        """
        If *wallet* is None we fabricate an in-memory Keypair so bittensor
        never tries to read ~/.bittensor keyfiles during tests.
        """
        # Always use an in-memory keypair; we never want bittensor to load
        # key-files from disk inside unit-tests, even if a wallet object is
        # supplied.

        self._session = None  # Must exist before parent ``__init__``.

        keypair = Keypair.create_from_seed(os.urandom(32))

        super().__init__(keypair)

        # Keep the placeholder value; real sessions are opened lazily by
        # bittensor when needed.  For tests we never open network sessions.
        self._session = None

        # Default timing window (seconds) used by forward()
        self.min_time = getattr(self, "min_time", 0.0)
        self.max_time = getattr(self, "max_time", 0.1)

    # ---------------------------------------------------------------------
    # Convenience helpers used by the local unit-tests
    # ---------------------------------------------------------------------

    def mock_response(self, synapse: bt.Synapse, data) -> bt.Synapse:
        """Return a successful mocked response embedding *data* in the synapse.

        For `IdentitySynapse` this sets the *variations* attribute, for
        `Dummy` it assigns *dummy_output* etc.  The method mimics a 200 status
        code so the caller can use regular assertions found in the test-suite.
        """
        synapse = synapse.copy()

        # Set dendrite metadata similar to the forward() path
        dendrite_info = bt.TerminalInfo()
        dendrite_info.status_code = 200
        dendrite_info.status_message = "OK"
        dendrite_info.process_time = 0.0
        synapse.dendrite = dendrite_info

        # Populate payload depending on synapse type
        if hasattr(synapse, "variations"):
            synapse.variations = data  # type: ignore[attr-defined]
        elif hasattr(synapse, "dummy_input"):
            # Expect *data* to be numeric output or None
            synapse.dummy_output = (
                data if data is not None else synapse.dummy_input * 2  # type: ignore[attr-defined]
            )

        return synapse

    def mock_timeout_response(self, synapse: bt.Synapse) -> bt.Synapse:
        """Return a timed-out synapse (HTTP 408)."""
        synapse = synapse.copy()
        dendrite_info = bt.TerminalInfo()
        dendrite_info.status_code = 408
        dendrite_info.status_message = "Timeout"
        dendrite_info.process_time = 999.0
        synapse.dendrite = dendrite_info
        return synapse

    def mock_error_response(self, synapse: bt.Synapse) -> bt.Synapse:
        """Return an internal-error synapse (HTTP 500)."""
        synapse = synapse.copy()
        dendrite_info = bt.TerminalInfo()
        dendrite_info.status_code = 500
        dendrite_info.status_message = "Internal Server Error"
        dendrite_info.process_time = 0.01
        synapse.dendrite = dendrite_info
        return synapse

    def mock_batch_responses(self, synapse_template: bt.Synapse, batch_data):
        """Generate a list of mocked responses given *batch_data* list."""
        responses = []
        for data in batch_data:
            responses.append(self.mock_response(synapse_template.copy(), data))
        return responses

    async def forward(
        self,
        axons: List[bt.axon],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = False,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i, axon):
                """Queries a single axon for a response."""

                start_time = time.time()
                s = synapse.copy()
                # Attach some more required data so it looks real
                s = self.preprocess_synapse_for_request(axon, s, timeout)
                # We just want to mock the response, so we'll just fill in some data
                process_time = random.uniform(self.min_time, self.max_time)
                # Success path (finished before timeout)
                if process_time < timeout:
                    s.dendrite.process_time = process_time
                    s.dummy_output = s.dummy_input * 2  # type: ignore[attr-defined]
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                else:
                    # Timed-out path
                    s.dummy_output = s.dummy_input  # Echo back original input
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"
                    s.dendrite.process_time = timeout

                return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "MockDendrite({})".format(self.keypair.ss58_address)

    # ------------------------------------------------------------------
    # Bittensor >=7.4.x destructor looks for an internal aiohttp session.
    # In our offline mocks we never open a session, so we override __del__ to
    # guard against AttributeError while still attempting a graceful close.
    # ------------------------------------------------------------------
    def __del__(self):  # noqa: D401 – simple destructor
        try:
            # Close session if the parent class created one.
            if hasattr(self, "close_session"):
                self.close_session()
        except Exception:
            # Silently ignore – unit-tests only care that no exception leaks.
            pass

# No longer monkey-patch ``bittensor.config`` – the original class is required
# for internal ``isinstance(..., bittensor.config)`` checks inside bittensor.
class _ConfigCompat(bt.config):
    """Subclass of bittensor.config that tolerates the *withconfig* kwarg."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("withconfig", None)
        super().__init__(*args, **kwargs)

        from types import SimpleNamespace

        if getattr(self, "neuron", None) is None:
            self.neuron = SimpleNamespace()


# Replace the reference so tests can call ``bt.config(withconfig=True)`` while
# preserving the fact that ``bt.config`` is still a *type* (avoiding
# isinstance() errors).
bt.config = _ConfigCompat