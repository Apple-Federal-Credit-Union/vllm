# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe a NIXL side-channel socket for handshake metadata readiness.

Authenticates with ``VLLM_NIXL_SIDE_CHANNEL_KEY`` like a decode worker does.
"""

import argparse

import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import GET_META_MSG
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_side_channel_keys,
    zmq_ctx,
)
from vllm.utils.network_utils import make_zmq_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--pp-rank", default=0, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--timeout-ms", default=1000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = get_side_channel_keys()
    if keys is None:
        raise SystemExit("VLLM_NIXL_SIDE_CHANNEL_KEY is not set")
    path = make_zmq_path("tcp", args.host, args.port)
    with zmq_ctx(zmq.REQ, path, keys) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
        sock.send(msgspec.msgpack.encode((GET_META_MSG, args.pp_rank, args.rank)))
        sock.recv()


if __name__ == "__main__":
    main()
