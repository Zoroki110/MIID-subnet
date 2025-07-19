import time

import asyncio
import random
import bittensor as bt

from typing import List


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)

        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock neurons who will be miners
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey=f"miner-hotkey-{i}",
                coldkey="mock-coldkey",
                balance=100000,
                stake=100000,
            )


class MockMetagraph(bt.metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")


class MockDendrite(bt.dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static response for all axons that are passed and adds some random delay.
    """

    def __init__(self, wallet):
        super().__init__(wallet)

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
        deserialize: bool = True,
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
                process_time = random.random()
                if process_time < timeout:
                    s.dendrite.process_time = str(time.time() - start_time)
                    # Update the status code and status message of the dendrite to match the axon
                    # TODO (developer): replace with your own expected synapse data
                    s.dummy_output = s.dummy_input * 2
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(process_time)
                else:
                    s.dummy_output = 0
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"
                    synapse.dendrite.process_time = str(timeout)

                # Return the updated synapse object after deserializing if requested
                if deserialize:
                    return s.deserialize()
                else:
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
