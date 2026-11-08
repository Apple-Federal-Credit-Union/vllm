# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-effect target for store payloads.

The weight-transfer tests pickle a reducer that names this function. If the
worker unpickles attacker bytes, the call is recorded here.
"""

RAN: list[str] = []


def note(token: str, pad: str = "") -> str:
    """Record that a store payload was interpreted."""
    del pad
    RAN.append(token)
    return token
