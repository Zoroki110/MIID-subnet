import random
import bittensor as bt
import numpy as np
from typing import List


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    bt.logging.warning(f"#########################################Metagraph: {metagraph}#########################################")
    bt.logging.warning(f"#########################################Metagraph type: {type(metagraph)}#########################################")
    bt.logging.warning(f"#########################################Metagraph n: {metagraph.n.item()}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons: {metagraph.axons}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons type: {type(metagraph.axons)}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons[uid]: {metagraph.axons[uid]}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons[uid] type: {type(metagraph.axons[uid])}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons[uid] is serving: {metagraph.axons[uid].is_serving}#########################################")
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []
    # Some test-suites pass a unittest.TestCase instance whose `metagraph`
    # attribute may be missing.  In that case fall back to
    # `self.neuron.metagraph` which is defined by the setUp fixture.

    metagraph = getattr(self, "metagraph", None)
    if metagraph is None and hasattr(self, "neuron"):
        metagraph = self.neuron.metagraph

    if metagraph is None:
        raise AttributeError("get_random_uids() expected object with metagraph or neuron.metagraph")

    vpermit_limit = 1024
    config_obj = getattr(self, "config", getattr(self, "neuron", None) and getattr(self.neuron, "config", None))
    if config_obj and hasattr(config_obj, "neuron"):
        vpermit_limit = getattr(config_obj.neuron, "vpermit_tao_limit", 1024)

    total_uids = len(getattr(metagraph, "axons", []))

    for uid in range(total_uids):
        uid_is_available = check_uid_availability(
            metagraph, uid, vpermit_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    bt.logging.info(f"#########################################Candidate uids: {candidate_uids}#########################################")
    bt.logging.info(f"#########################################Available uids: {avail_uids}#########################################")
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))
    bt.logging.info(f"#########################################K: {k}#########################################")
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    bt.logging.info(f"#########################################Available uids: {available_uids}#########################################")
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    bt.logging.info(f"#########################################Available uids: {available_uids}#########################################")
    uids = np.array(random.sample(available_uids, k))
    bt.logging.info(f"#########################################Uids: {uids}#########################################")
    return uids
