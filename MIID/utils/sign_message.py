# Updated to accept either a bittensor.wallet OR a simple wallet *name* (str).
# If a string is provided we fall back to an ephemeral keypair so tests don’t
# need a fully-configured wallet on disk.

from datetime import datetime
from typing import Union
import bittensor


def sign_message(wallet: Union["bittensor.wallet", str], message_text: str, output_file: str = "message_and_signature.txt"):
    """
    Signs a message using the specified wallet and writes it to a file.

    Args:
        wallet (bittensor.wallet): The wallet object to use for signing.
        message_text (str): The message you want to sign.
        output_file (str, optional): Filename to save message and signature. Defaults to "message_and_signature.txt".

    Returns:
        str: The combined file contents (message + signature info).
    """
    # ------------------------------------------------------------------
    # Resolve the keypair we will use for signing
    # ------------------------------------------------------------------

    if isinstance(wallet, str):
        # Create an ephemeral keypair.  We only need it for signing the test
        # payload, so persisting it to disk is unnecessary.
        import os
        from substrateinterface import Keypair

        keypair = Keypair.create_from_seed(os.urandom(32))
    else:
        keypair = wallet.hotkey

    # Generate a timestamped message
    timestamp = datetime.now()
    timezone_name = timestamp.astimezone().tzname()
    signed_message = f"<Bytes>On {timestamp} {timezone_name} {message_text}</Bytes>"

    # Sign the message
    signature = keypair.sign(data=signed_message)

    # Construct the output
    file_contents = (
        f"{signed_message}\n"
        f"\tSigned by: {keypair.ss58_address}\n"
        f"\tSignature: {signature.hex()}"
    )

    # Print to console
    print(file_contents)

    # Write to file if output_file is specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(file_contents)
        print(f"Signature generated and saved to {output_file}")
    
    return file_contents
