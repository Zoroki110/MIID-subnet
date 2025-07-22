from types import SimpleNamespace

class MockWallet:
    def __init__(self, config=None, name="mock", hotkey="mock"):
        self.name = name
        self.hotkey = SimpleNamespace(ss58_address=f"{hotkey}_hotkey")
        self.coldkey = SimpleNamespace(ss58_address=f"{name}_coldkey")


def get_mock_wallet(*args, **kwargs):
    return MockWallet()

def get_mock_hotkey(uid=None):
    return f"hotkey_{uid if uid is not None else 'mock'}"

def get_mock_coldkey(uid=None):
    return f"coldkey_{uid if uid is not None else 'mock'}"

def get_mock_keypair(uid=None):
    return SimpleNamespace(ss58_address=f"keypair_{uid if uid is not None else 'mock'}")
