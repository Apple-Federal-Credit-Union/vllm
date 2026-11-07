# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P KV cache sharing protocol constants and documentation.

Protocol Overview
=================

Two session types communicate over a bidirectional message channel:
- P2PClientSession (requests blocks from a server)
- P2PServerSession (serves blocks to a client)

Connection Lifecycle
--------------------

1. Client opens a connection and sends ConnectMsg with its identity,
   RDMA metadata, memory layout, and config fingerprint.
2. Server validates block_len and config_fingerprint, registers the
   RDMA peer, and replies with ConnectAckMsg.
3. Client receives ConnectAckMsg and transitions to ready state
   (flushes any queued messages).
4. Either side may send DisconnectMsg to gracefully close.

Block Transfer Flow (happy path)
---------------------------------

1. Client sends FetchMsg with a kv_request_id and lists of
   block keys + remote indexes where it wants the data written.
   A request may run several lookup→fetch rounds; symmetric-P2P
   messages carry ROUND_SEQ so each round's supply, demand, and
   completion stay isolated. The terminal empty FetchMsg is the
   server-side "request finished" signal for the id: parked
   LookupMsg batches are popped and ``cb.finish_request`` fires on
   each.
2. Server matches requested blocks against locally stored blocks:
   - Blocks already available are transferred immediately via RDMA.
   - Blocks not yet available are recorded as "demanded" and
     transferred when the server later stores them.
3. When all blocks for a kv_request_id are transferred, the server
   sends TransferDoneMsg (success=True) to the client.
4. Client reports the load job as complete.

Abort Flow (timeout path)
--------------------------

1. If the client times out waiting for TransferDoneMsg, it sends
   AbortFetchMsg to cancel the request.
2. Server cancels inflight transfers for that kv_request_id and
   replies with AbortAckMsg.
3. Client receives AbortAckMsg and reports the load job as failed.
4. If AbortAckMsg itself times out, the client fails the job anyway.

Message Format
--------------

All messages are dicts serialized with msgpack. Every message has a
TYPE_KEY key identifying its type. Additional fields depend on the
message type (see per-message class docstrings below).

Security
--------

The control socket is reachable by any process that can connect to it,
so peers authenticate with a pre-shared key (``VLLM_P2P_SIDE_CHANNEL_PSK``)
before any KV is exposed. The dialer sends its ConnectMsg first with an
AUTH tag (see :func:`compute_control_auth`). The acceptor stays silent and
verifies that tag before it discloses its own agent metadata or hash seed,
registers the peer's RDMA memory, or serves a block. With no key configured
an acceptor cannot authenticate anyone and refuses every inbound peer.

Sessions wrap all incoming message handling in try/except to guard
against malformed messages from adversarial or buggy peers. Invalid
messages are logged and dropped without crashing the session.

The config_fingerprint field in ConnectMsg ensures peers have
compatible model configurations (model, dtype, block sizes). When a side
has a fingerprint, the peer must present the same non-empty value; an
omitted fingerprint is rejected during the handshake.
"""

import hashlib
import hmac

TYPE_KEY = "type"

# Domain-separation label for the control-channel authentication tag, so a
# proof minted here can never be confused with any other HMAC the process
# computes under the same pre-shared key.
_AUTH_LABEL = b"vllm-p2p-control-auth-v1"


def compute_control_auth(psk: str, sender_id: str, receiver_id: str) -> bytes:
    """Authentication tag proving the sender holds the pre-shared key.

    The tag is bound to the direction of one message: a side sends
    ``compute_control_auth(psk, self, peer)`` and requires the peer to send
    ``compute_control_auth(psk, peer, self)``. The two values differ, so a
    peer that echoes our own tag back is rejected, and the tag a side sends
    is independent of which side dialed. Binding the ZMQ identities means a
    tag minted for one peer pair cannot be presented to another.

    Args:
        psk: The shared pre-shared key (``VLLM_P2P_SIDE_CHANNEL_PSK``).
        sender_id: ``host:port`` identity of the side sending the tag.
        receiver_id: ``host:port`` identity of the side verifying it.

    Returns:
        The HMAC-SHA256 tag for this sender and receiver.

    """
    msg = b"\0".join(
        (_AUTH_LABEL, sender_id.encode("utf-8"), receiver_id.encode("utf-8"))
    )
    return hmac.new(psk.encode("utf-8"), msg, hashlib.sha256).digest()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require(msg: dict, key: str, typ: type, *, name: str = "") -> None:
    """Raise ValueError if msg[key] is missing or not isinstance(typ)."""
    val = msg.get(key)
    if not isinstance(val, typ):
        label = name or key
        raise ValueError(f"{label}: expected {typ.__name__}, got {type(val).__name__}")


def _require_pos_int(msg: dict, key: str, *, name: str = "") -> None:
    """Raise ValueError if msg[key] is not a positive int."""
    val = msg.get(key)
    if not isinstance(val, int) or val <= 0:
        label = name or key
        raise ValueError(f"{label}: expected positive int, got {val!r}")


def _require_non_neg_int(msg: dict, key: str, *, name: str = "") -> None:
    """Raise ValueError if msg[key] is not a non-negative int."""
    val = msg.get(key)
    if not isinstance(val, int) or val < 0:
        label = name or key
        raise ValueError(f"{label}: expected non-negative int, got {val!r}")


def _require_list(msg: dict, key: str, *, name: str = "") -> None:
    """Raise ValueError if msg[key] is not a list."""
    val = msg.get(key)
    if not isinstance(val, list):
        label = name or key
        raise ValueError(f"{label}: expected list, got {type(val).__name__}")


# ---------------------------------------------------------------------------
# Message classes
# ---------------------------------------------------------------------------


class ConnectMsg:
    """Client → Server: initial handshake request.

    Fields:
        PEER_ID: Local peer identity string.
        AGENT_METADATA: RDMA agent metadata (opaque bytes).
        BASE_ADDR: Base memory address of the KV block region.
        NUM_BLOCKS: Number of blocks in the KV block region.
        BLOCK_LEN: Size in bytes of each block (must match between peers).
        CONFIG_FINGERPRINT: SHA-256 prefix of the model configuration.
            Peers with different fingerprints are incompatible.
        HASH_SEED: The peer's effective prefix-cache hash seed (PYTHONHASHSEED
            if set, otherwise the built-in default). Block hashes chain from a
            seed derived from it, so peers with different values compute
            different hashes for identical content and must not exchange blocks.
        AUTH: Pre-shared-key authentication tag for this peer pair (see
            :func:`compute_control_auth`). Proves the sender holds the shared
            ``VLLM_P2P_SIDE_CHANNEL_PSK`` without disclosing it. The acceptor
            verifies it before registering the peer, serving any block, or
            disclosing its own agent metadata and hash seed.
    """

    TYPE = "connect"
    PEER_ID = "peer_id"
    AGENT_METADATA = "agent_metadata"
    BASE_ADDR = "base_addr"
    NUM_BLOCKS = "num_blocks"
    BLOCK_LEN = "block_len"
    CONFIG_FINGERPRINT = "config_fingerprint"
    HASH_SEED = "hash_seed"
    AUTH = "auth"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, ConnectMsg.PEER_ID, str)
        _require(msg, ConnectMsg.AGENT_METADATA, bytes)
        _require_non_neg_int(msg, ConnectMsg.BASE_ADDR)
        _require_pos_int(msg, ConnectMsg.NUM_BLOCKS)
        _require_pos_int(msg, ConnectMsg.BLOCK_LEN)
        _require(msg, ConnectMsg.HASH_SEED, str)


class ConnectAckMsg:
    """Server → Client: handshake acknowledgement.

    Fields:
        PEER_ID: Server's peer identity string.
    """

    TYPE = "connect_ack"
    PEER_ID = "peer_id"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, ConnectAckMsg.PEER_ID, str)


class DisconnectMsg:
    """Either → Either: graceful connection close.

    No additional fields beyond TYPE_KEY.
    """

    TYPE = "disconnect"


class FetchMsg:
    """Client → Server: request blocks for one lookup round.

    A non-empty fetch closes only its round. The terminal empty FetchMsg
    (``KEYS`` and ``BLOCK_INDEXES`` both empty) is the "request
    finished" signal: the server pops parked LookupMsg batches, calls
    ``cb.finish_request`` on each, and drains any leftover supply.

    Fields:
        KV_REQUEST_ID: Identifies this block transfer request.
        KEYS: List of block keys (OffloadKey bytes). May be empty.
        BLOCK_INDEXES: List of remote block indexes (same length as KEYS).
        ROUND_SEQ: Lookup round this fetch closes. PD clients never probe
            and stay on their single round 0.
    """

    TYPE = "fetch"
    KV_REQUEST_ID = "kv_request_id"
    KEYS = "keys"
    BLOCK_INDEXES = "block_indexes"
    ROUND_SEQ = "round_seq"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, FetchMsg.KV_REQUEST_ID, str)
        _require_non_neg_int(msg, FetchMsg.ROUND_SEQ)
        _require_list(msg, FetchMsg.KEYS)
        _require_list(msg, FetchMsg.BLOCK_INDEXES)
        keys = msg[FetchMsg.KEYS]
        indexes = msg[FetchMsg.BLOCK_INDEXES]
        if len(keys) != len(indexes):
            raise ValueError(
                f"keys/block_indexes length mismatch: {len(keys)} vs {len(indexes)}"
            )
        for idx in indexes:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"block_indexes: invalid index {idx!r}")


class LookupMsg:
    """Client → Server: probe which block keys the peer holds.

    Sent on the consumer side under symmetric P2P (do_p2p_fetch=true)
    after the consumer has aggregated per-block lookups across a
    scheduler step. The producer replies with one or more LookupRespMsg
    covering the requested keys.

    Fields:
        KV_REQUEST_ID: Identifies this lookup transaction.
        KEYS: List of block keys (OffloadKey bytes) to probe.
        ROUND_SEQ: Lookup round these probes belong to; pinned supply is
            parked under it for that round's fetch.
    """

    TYPE = "lookup"
    KV_REQUEST_ID = "kv_request_id"
    KEYS = "keys"
    ROUND_SEQ = "round_seq"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, LookupMsg.KV_REQUEST_ID, str)
        _require_list(msg, LookupMsg.KEYS)
        _require_non_neg_int(msg, LookupMsg.ROUND_SEQ)


class LookupRespMsg:
    """Server → Client: per-key hit/miss answer for a prior LookupMsg.

    Carries two parallel arrays of equal length so each (key,
    hit) pair is self-describing. The producer is free to split or
    coalesce responses across multiple LookupMsgs for the same
    KV_REQUEST_ID — the consumer matches each pair back to its
    pending entry by (KV_REQUEST_ID, key).

    Fields:
        KV_REQUEST_ID: The lookup transaction this responds to.
        KEYS: List of block keys answered by this message.
        HITS: Parallel list of bools — True if the producer holds the
            corresponding block, False otherwise.
    """

    TYPE = "lookup_resp"
    KV_REQUEST_ID = "kv_request_id"
    KEYS = "keys"
    HITS = "hits"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, LookupRespMsg.KV_REQUEST_ID, str)
        _require_list(msg, LookupRespMsg.KEYS)
        _require_list(msg, LookupRespMsg.HITS)
        keys = msg[LookupRespMsg.KEYS]
        hits = msg[LookupRespMsg.HITS]
        if len(keys) != len(hits):
            raise ValueError(f"keys/hits length mismatch: {len(keys)} vs {len(hits)}")
        for hit in hits:
            if not isinstance(hit, bool):
                raise ValueError(f"hits: invalid value {hit!r}")


class TransferDoneMsg:
    """Server → Client: all blocks transferred for a request.

    Fields:
        KV_REQUEST_ID: The request that completed.
        SUCCESS: Whether the transfer completed successfully.
        ROUND_SEQ: The fetch round that completed. Several loads can be
            in flight per id (the scheduler submits loads incrementally),
            so completions are matched by round.
    """

    TYPE = "transfer_done"
    KV_REQUEST_ID = "kv_request_id"
    SUCCESS = "success"
    ROUND_SEQ = "round_seq"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, TransferDoneMsg.KV_REQUEST_ID, str)
        _require(msg, TransferDoneMsg.SUCCESS, bool)
        _require_non_neg_int(msg, TransferDoneMsg.ROUND_SEQ)


class AbortFetchMsg:
    """Client → Server: cancel a pending request.

    Fields:
        KV_REQUEST_ID: The request to cancel.
        ROUND_SEQ: The fetch round to cancel.
    """

    TYPE = "abort_fetch"
    KV_REQUEST_ID = "kv_request_id"
    ROUND_SEQ = "round_seq"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, AbortFetchMsg.KV_REQUEST_ID, str)
        _require_non_neg_int(msg, AbortFetchMsg.ROUND_SEQ)


class AbortAckMsg:
    """Server → Client: acknowledge cancellation.

    Fields:
        KV_REQUEST_ID: The request that was cancelled.
        ROUND_SEQ: The round that was cancelled; echoes AbortFetchMsg.
    """

    TYPE = "abort_ack"
    KV_REQUEST_ID = "kv_request_id"
    ROUND_SEQ = "round_seq"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, AbortAckMsg.KV_REQUEST_ID, str)
        _require_non_neg_int(msg, AbortAckMsg.ROUND_SEQ)
