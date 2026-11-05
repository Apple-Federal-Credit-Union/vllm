# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF3FS Metadata Server with key-based organization."""

import argparse
import hmac
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import orjson
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import vllm.envs as envs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TOKEN_ENV = "VLLM_HF3FS_METADATA_TOKEN"


def _is_page_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _load_metadata_token() -> str:
    token = envs.VLLM_HF3FS_METADATA_TOKEN
    if not token:
        raise ValueError(
            f"{TOKEN_ENV} must be set to the shared HF3FS metadata server token"
        )
    return token


@dataclass
class RankFileMetadata:
    """Manages file page allocation for a single rank."""

    rank_id: int
    num_pages: int
    free_pages: list[int]

    def allocate_pages(self, num_pages: int) -> list[int]:
        """Allocate specified number of free pages."""
        if len(self.free_pages) < num_pages:
            return []

        allocated = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        return allocated

    def release_pages(self, page_indices: list[int]) -> None:
        """Release pages back to free pool."""
        for page_idx in page_indices:
            if page_idx not in self.free_pages:
                self.free_pages.append(page_idx)

    def get_free_page_count(self) -> int:
        """Get current number of free pages."""
        return len(self.free_pages)


@dataclass
class KeyMetadata:
    """Manages metadata for a single key across multiple ranks."""

    key: str
    rank_to_page: dict[int, int]  # rank -> allocated page index
    tp_world_size: int

    def add_rank_page(self, rank: int, page_index: int) -> None:
        """Add page allocation for a specific rank."""
        self.rank_to_page[rank] = page_index

    def get_all_pages(self) -> list[tuple[int, int]]:
        """Get all (rank, page) pairs for this key."""
        return [(rank, page) for rank, page in self.rank_to_page.items()]

    def get_rank_page(self, rank: int) -> int | None:
        """Get page index for a specific rank."""
        return self.rank_to_page.get(rank)

    def is_complete(self) -> bool:
        """Check if all ranks in the TP world have allocated pages."""
        return len(self.rank_to_page) == self.tp_world_size


class GlobalMetadataState:
    """Manages global metadata state across all ranks and keys."""

    def __init__(self):
        self.global_lock = threading.RLock()
        self.rank_metadata: dict[int, RankFileMetadata] = {}
        self.key_metadata: dict[str, KeyMetadata] = {}
        # rank -> page -> key for pages allocated but not yet confirmed.
        self.pending_pages: dict[int, dict[int, str]] = {}

    def clear(self) -> None:
        """Clear all metadata state."""
        with self.global_lock:
            self.rank_metadata.clear()
            self.key_metadata.clear()
            self.pending_pages.clear()
            logger.info("Cleared all metadata state")

    def initialize_rank(self, rank: int, num_pages: int) -> None:
        """Initialize a new rank with specified number of pages."""
        with self.global_lock:
            if rank not in self.rank_metadata:
                self.rank_metadata[rank] = RankFileMetadata(
                    rank, num_pages, list(range(num_pages))
                )
                logger.info("Initialized rank %s with %s pages", rank, num_pages)

    def allocate_pages_for_keys(
        self, rank: int, keys: list[tuple[str, str]]
    ) -> dict[str, int]:
        """Allocate one page for each key on the specified rank.

        Args:
            rank: Rank ID to allocate pages on
            keys: List of keys to allocate pages for

        Returns:
            Dictionary mapping key -> allocated page index

        """
        with self.global_lock:
            if rank not in self.rank_metadata:
                raise ValueError(f"Rank {rank} not initialized")
            if not all(
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], str)
                for item in keys
            ):
                raise ValueError("Keys must be (key, prefix_key) pairs")

            # Batch allocate pages for all keys
            num_pages_needed = len(keys)
            allocated_pages = self.rank_metadata[rank].allocate_pages(num_pages_needed)

            if len(allocated_pages) < num_pages_needed:
                logger.warning(
                    "Rank %s only allocated %s pages for %s keys",
                    rank,
                    len(allocated_pages),
                    num_pages_needed,
                )

            pending = self.pending_pages.setdefault(rank, {})
            allocation_results = {}
            for i, (key, prefix_key) in enumerate(keys):
                if key in self.key_metadata:
                    key_meta = self.key_metadata[key]
                    if key_meta.is_complete() and rank in key_meta.rank_to_page:
                        # key is already fully written, reuse the existing page
                        # and release the allocated pages back to the free pool.
                        if i < len(allocated_pages):
                            self.rank_metadata[rank].release_pages([allocated_pages[i]])
                        allocation_results[key] = key_meta.rank_to_page[rank]
                        continue

                if i < len(allocated_pages):
                    allocation_results[key] = allocated_pages[i]
                    pending[allocated_pages[i]] = key
                else:
                    allocation_results[key] = -1  # No pages available

            return allocation_results

    def confirm_write_for_keys(
        self,
        rank: int,
        key_confirmations: list[tuple[str, int]],
        pages_to_release: list[int] | None = None,
    ) -> None:
        """Confirm write operations for keys and update metadata.

        A confirmation is accepted only for the page this rank was allocated
        for that key, or for the page the key is already bound to on this rank.
        The whole batch is rejected if any confirmation is not. Only pages that
        are allocated on this rank and not yet confirmed can be released.

        Args:
            rank: Rank ID that confirmed the writes
            key_confirmations: List of (key, page_index) tuples
            pages_to_release: List of page indices to release back to free pool

        Raises:
            ValueError: If the rank is not initialized or the input is malformed.
            PermissionError: If a confirmation names a page that was not
                allocated to that key on this rank.

        """
        with self.global_lock:
            if rank not in self.rank_metadata:
                raise ValueError(f"Rank {rank} not initialized")
            pending = self.pending_pages.setdefault(rank, {})

            for confirmation in key_confirmations:
                if (
                    not isinstance(confirmation, (list, tuple))
                    or len(confirmation) != 2
                ):
                    raise ValueError("Confirmations must be (key, page_index) pairs")
                key, page_index = confirmation
                if not isinstance(key, str) or not _is_page_index(page_index):
                    raise ValueError("Confirmations must be (key, page_index) pairs")
                key_meta = self.key_metadata.get(key)
                bound_page = key_meta.get_rank_page(rank) if key_meta else None
                if bound_page != page_index and pending.get(page_index) != key:
                    raise PermissionError(
                        f"Page {page_index} on rank {rank} was not allocated "
                        "for this key"
                    )

            for key, page_index in key_confirmations:
                if key not in self.key_metadata:
                    # Need to determine tp_world_size from rank_metadata
                    tp_world_size = len(self.rank_metadata)
                    self.key_metadata[key] = KeyMetadata(key, {}, tp_world_size)
                key_meta = self.key_metadata[key]
                bound_page = key_meta.get_rank_page(rank)
                if bound_page == page_index:
                    continue
                del pending[page_index]
                if bound_page is None:
                    key_meta.add_rank_page(rank, page_index)
                else:
                    # A concurrent save already bound this key on this rank.
                    self.rank_metadata[rank].release_pages([page_index])

            released = []
            for page_index in pages_to_release or []:
                if _is_page_index(page_index) and page_index in pending:
                    del pending[page_index]
                    released.append(page_index)
                else:
                    logger.warning(
                        "Ignoring release of page %s on rank %s: not an "
                        "unconfirmed allocation",
                        page_index,
                        rank,
                    )
            if released:
                self.rank_metadata[rank].release_pages(released)
                logger.debug(
                    "Released %s pages on rank %s: %s",
                    len(released),
                    rank,
                    released,
                )

    def batch_key_exists(self, keys: list[str]) -> list[bool]:
        """Check if keys exist in metadata and all ranks have confirmed writes.

        Args:
            keys: List of keys to check

        Returns:
            List of boolean values indicating key existence and completion

        """
        with self.global_lock:
            results = []
            for key in keys:
                if key not in self.key_metadata:
                    results.append(False)
                else:
                    # Check if all ranks in the TP world have confirmed writes
                    key_meta = self.key_metadata[key]
                    results.append(key_meta.is_complete())
            return results

    def get_key_locations(self, rank: int, keys: list[str]) -> list[int | None]:
        """Get page indices for keys on a specific rank.

        Args:
            rank: Rank ID to query
            keys: List of keys to look up

        Returns:
            List of page indices in the same order as input keys (None if key not found)

        """
        with self.global_lock:
            if rank not in self.rank_metadata:
                raise ValueError(f"Rank {rank} not initialized")

            results = []
            for key in keys:
                if key in self.key_metadata:
                    key_meta = self.key_metadata[key]
                    if key_meta.is_complete():
                        page_index = key_meta.get_rank_page(rank)
                    else:
                        page_index = None

                    results.append(page_index)
                else:
                    results.append(None)

            return results


class Hf3fsMetadataServer:
    """HF3FS Metadata Server with improved key-based organization."""

    def __init__(self, persistence_path: str | None = None, save_interval: int = 60):
        self._expected_authorization = f"Bearer {_load_metadata_token()}".encode()
        self.state = GlobalMetadataState()
        self.app = FastAPI(
            default_response_class=ORJSONResponse,
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
        self.app.middleware("http")(self._authorize)
        self._setup_routes()

    async def _authorize(self, request: Request, call_next):
        """Reject any request that does not carry the shared bearer token."""
        provided = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(provided, self._expected_authorization):
            return ORJSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    async def _read_json(self, request: Request) -> dict:
        """Parse request JSON using orjson."""
        body = await request.body()
        return orjson.loads(body)

    def _setup_routes(self):
        """Setup FastAPI routes for new API design."""
        self.app.post("/rank/{rank}/initialize")(self.initialize_rank)
        self.app.post("/keys/batch_allocate")(self.batch_allocate_pages_for_keys)
        self.app.post("/keys/confirm_write")(self.confirm_write_for_keys)
        self.app.post("/keys/batch_exists")(self.batch_key_exists)
        self.app.post("/keys/get_locations")(self.get_key_locations)
        self.app.post("/clear")(self.clear)

    async def initialize_rank(self, rank: int, request: Request):
        """Initialize a rank with specified number of pages."""
        data = await self._read_json(request)
        role = data.get("role", "worker")
        num_pages = data.get("num_pages", 0)

        if role == "scheduler":
            return ORJSONResponse(
                {"message": "Scheduler role does not require initialization"}
            )

        if role == "worker" and num_pages > 0:
            self.state.initialize_rank(rank, num_pages)
            return ORJSONResponse(
                {"message": f"Rank {rank} initialized with {num_pages} pages"}
            )
        else:
            raise HTTPException(
                status_code=400, detail="Invalid initialization parameters"
            )

    async def batch_allocate_pages_for_keys(self, request: Request):
        """Allocate one page for each key on a specific rank."""
        data = await self._read_json(request)
        rank = data.get("rank")
        keys = data.get("keys", [])

        # Validate input format
        if rank is None or not isinstance(keys, list):
            raise HTTPException(
                status_code=400, detail="Invalid request format: need 'rank' and 'keys'"
            )

        try:
            # Perform allocation
            results = self.state.allocate_pages_for_keys(rank, keys)

            # Convert results to response format
            response = {"rank": rank, "results": list(results.items())}
            return ORJSONResponse(response)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Allocation failed: {str(e)}"
            ) from e

    async def confirm_write_for_keys(self, request: Request):
        """Confirm write operations for keys."""
        data = await self._read_json(request)
        rank = data.get("rank")
        confirmations = data.get("confirmations", [])
        pages_to_release = data.get("pages_to_release", [])

        # Validate input format
        if (
            rank is None
            or not isinstance(confirmations, list)
            or not isinstance(pages_to_release, list)
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid request format: need 'rank' and 'confirmations'",
            )

        try:
            self.state.confirm_write_for_keys(rank, confirmations, pages_to_release)

            return Response(status_code=204)

        except PermissionError as e:
            logger.warning("Rejected write confirmation on rank %s: %s", rank, e)
            raise HTTPException(status_code=403, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("Confirm write for keys failed: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Confirmation failed: {str(e)}"
            ) from e

    async def batch_key_exists(self, request: Request):
        """Check if multiple keys exist in metadata."""
        data = await self._read_json(request)
        keys = data.get("keys", [])

        if not isinstance(keys, list):
            raise HTTPException(status_code=400, detail="Invalid keys format")

        try:
            exists_results = self.state.batch_key_exists(keys)
            return ORJSONResponse({"exists": exists_results})
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Key existence check failed: {str(e)}"
            ) from e

    async def get_key_locations(self, request: Request):
        """Get page indices for keys on a specific rank."""
        data = await self._read_json(request)
        rank = data.get("rank")
        keys = data.get("keys", [])

        # Validate input format
        if rank is None or not isinstance(keys, list):
            raise HTTPException(
                status_code=400, detail="Invalid request format: need 'rank' and 'keys'"
            )

        try:
            # Get key locations
            locations = self.state.get_key_locations(rank, keys)
            return ORJSONResponse({"locations": locations})
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to get key locations: {str(e)}"
            ) from e

    async def clear(self, request: Request):
        """Clear the metadata server."""
        self.state.clear()
        return Response(status_code=204)

    def run(self, host: str = "127.0.0.1", port: int = 18000):
        """Run the metadata server."""
        import uvicorn

        logger.info("Starting improved metadata server on http://%s:%s", host, port)
        uvicorn.run(self.app, host=host, port=port)


# --- Client implementation ---
class Hf3fsMetadataInterface(ABC):
    """Interface for HF3FS metadata operations."""

    @abstractmethod
    def initialize(self, rank: int, num_pages: int = 0, role: str = "worker") -> None:
        """Initialize the metadata service with specified number of pages."""
        pass

    @abstractmethod
    def allocate_pages_for_keys(
        self, rank: int, keys: list[tuple[str, str]]
    ) -> list[tuple[str, int]]:
        """Allocate one page for each key on the specified rank."""
        pass

    @abstractmethod
    def confirm_write_for_keys(
        self,
        rank: int,
        key_confirmations: list[tuple[str, int]],
        pages_to_release: list[int] | None = None,
    ) -> None:
        """Confirm write operations for keys and optionally release pages."""
        pass

    @abstractmethod
    def batch_key_exists(self, keys: list[str]) -> list[bool]:
        """Check if keys exist and are complete across all ranks."""
        pass

    @abstractmethod
    def get_key_locations(self, rank: int, keys: list[str]) -> list[int]:
        """Get page indices for keys on a specific rank."""
        pass


class Hf3fsGlobalMetadataClient(Hf3fsMetadataInterface):
    """Global HTTP metadata client for HF3FS."""

    def __init__(self, base_url: str = "http://localhost:18000", max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {_load_metadata_token()}"

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)

    def _post(self, endpoint: str, json_data: dict) -> dict:
        """Make POST request to metadata server."""
        try:
            url = f"{self.base_url}/{endpoint}"
            headers = {"Content-Type": "application/json"}
            payload = orjson.dumps(json_data)
            response = self._session.post(url, data=payload, headers=headers)
            response.raise_for_status()

            if response.status_code == 204 or not response.content:
                return {}
            return orjson.loads(response.content)
        except requests.exceptions.RequestException as e:
            logger.error("Failed to POST to %s after retries: %s", endpoint, e)
            raise RuntimeError(f"Failed to connect to metadata server: {e}") from e

    def initialize(self, rank: int, num_pages: int = 0, role: str = "worker") -> None:
        """Initialize a rank with specified number of pages."""
        self._post(f"rank/{rank}/initialize", {"num_pages": num_pages, "role": role})

    def allocate_pages_for_keys(
        self, rank: int, keys: list[tuple[str, str]]
    ) -> list[tuple[str, int]]:
        """Allocate pages for keys on the specified rank."""
        response = self._post("keys/batch_allocate", {"rank": rank, "keys": keys})

        # Convert response to expected format
        return response.get("results", {})

    def confirm_write_for_keys(
        self,
        rank: int,
        key_confirmations: list[tuple[str, int]],
        pages_to_release: list[int] | None = None,
    ) -> None:
        """Confirm write operations for keys and optionally release pages."""
        payload = {
            "rank": rank,
            "confirmations": key_confirmations,
            "pages_to_release": pages_to_release or [],
        }

        self._post("keys/confirm_write", payload)

    def batch_key_exists(self, keys: list[str]) -> list[bool]:
        """Check if keys exist and are complete across all ranks."""
        response = self._post("keys/batch_exists", {"keys": keys})
        return response.get("exists", [])

    def get_key_locations(self, rank: int, keys: list[str]) -> list[int]:
        """Get page indices for keys on a specific rank."""
        response = self._post("keys/get_locations", {"rank": rank, "keys": keys})
        return response.get("locations", [])


def run_metadata_server(
    host: str = "127.0.0.1",
    port: int = 18000,
):
    """Run the improved HF3FS metadata server."""
    server = Hf3fsMetadataServer()
    server.run(host=host, port=port)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Improved HF3FS Metadata Server. Requires the "
            f"{TOKEN_ENV} environment variable; clients must send it as a "
            "bearer token."
        )
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to. Use an internal address to serve "
        "other nodes.",
    )
    parser.add_argument(
        "--port", type=int, default=18000, help="Port to run the server on."
    )
    args = parser.parse_args()

    run_metadata_server(args.host, args.port)
