# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2025 Yanez

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

import typing
import bittensor as bt
import json
from typing import List, Dict, Optional

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class IdentitySynapse(bt.Synapse):
    """
    Protocol for requesting identity variations from miners.
    
    Attributes:
    - names: List of seed names to generate variations for
    - query_template: Template string for the LLM prompt with {name} placeholder
    - variations: Optional dictionary containing the generated variations for each name
    """
    
    # Required request input, filled by sending dendrite caller
    names: List[str]
    query_template: str

    timeout: float = 120.0
    
    # Optional request output, filled by receiving axon
    variations: Optional[Dict[str, List[str]]] = None
    
    def deserialize(self) -> Dict[str, List[str]]:
        """
        Deserialize the variations output.
        
        Returns:
        - Dict[str, List[str]]: Dictionary mapping each name to its list of variations
        """
        return self.variations


class Dummy(bt.Synapse):
    """A minimal synapse used by tests and examples.

    Attributes
    ----------
    dummy_input : int
        The integer sent by the requester.
    dummy_output : Optional[int]
        The integer returned by the responder, typically some deterministic
        function of ``dummy_input``.
    """
    dummy_input: int
    dummy_output: Optional[int] = None

    def deserialize(self) -> int:
        """Return the output in its native python type, defaulting to 0 if None."""
        return int(self.dummy_output) if self.dummy_output is not None else 0

    # ------------------------------------------------------------------
    # Helpers to make Dummy instances compare equal to their output value so
    # that unit-tests can assert ``response == expected_int`` regardless of
    # whether they received a Synapse or a raw int.
    # ------------------------------------------------------------------

    def __int__(self):  # type: ignore[override]
        return int(self.dummy_output) if self.dummy_output is not None else 0

    def __eq__(self, other):  # type: ignore[override]
        if isinstance(other, int):
            return self.dummy_output == other
        return super().__eq__(other)
