"""Node-wide, content-addressed cache of verified container image layers."""

import concurrent.futures
import contextlib
import fcntl
import hashlib
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.request
from typing import BinaryIO, Callable, Iterator, List, Optional

from ray.experimental.sandbox.exceptions import SandboxCreationError

logger = logging.getLogger(__name__)

BLOBS_SUBDIR = "blobs"
_LAYER_CACHE_MAX_BYTES_ENV = "RAY_SANDBOX_LAYER_CACHE_MAX_BYTES"
_PULL_PARALLELISM_ENV = "RAY_SANDBOX_LAYER_PULL_PARALLELISM"
_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")
_CHUNK = 1024 * 1024


def _blob_dir(images_dir: str) -> str:
    return os.path.join(images_dir, BLOBS_SUBDIR, "sha256")


def blob_path(images_dir: str, digest: str) -> str:
    """Return the cached path for a SHA-256 layer digest."""
    match = _DIGEST.fullmatch(digest)
    if not match:
        raise SandboxCreationError(f"Unsupported layer digest {digest!r}")
    return os.path.join(_blob_dir(images_dir), match.group(1))


def layer_cache_max_bytes(images_dir: str) -> int:
    """Return the layer cache cap, defaulting to a quarter of its filesystem."""
    raw = os.environ.get(_LAYER_CACHE_MAX_BYTES_ENV)
    if raw is not None and raw.strip():
        try:
            return max(int(raw), 0)
        except ValueError:
            logger.warning(
                "Ignoring %s=%r: expected an integer number of bytes.",
                _LAYER_CACHE_MAX_BYTES_ENV,
                raw,
            )
    try:
        return shutil.disk_usage(images_dir).total // 4
    except OSError as err:
        logger.warning("Failed to determine sandbox layer cache capacity: %s", err)
        return 0


def _download(
    open_blob: Callable[[], urllib.request.Request],
    digest: str,
    path: str,
    timeout: float,
    stop: threading.Event,
) -> None:
    """Verify a temporary download before publishing it under its digest."""
    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(path), prefix=".partial-", delete=False
    ) as partial:
        try:
            hasher = hashlib.sha256()
            with urllib.request.urlopen(open_blob(), timeout=timeout) as response:
                while not stop.is_set():
                    chunk = response.read1(_CHUNK)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    partial.write(chunk)
            if stop.is_set():
                raise concurrent.futures.CancelledError()
            if hasher.hexdigest() != digest.removeprefix("sha256:"):
                raise SandboxCreationError(
                    f"Layer {digest} failed verification: got sha256:{hasher.hexdigest()}"
                )
            partial.flush()
            os.fchmod(partial.fileno(), 0o644)
            os.replace(partial.name, path)
        except BaseException:
            os.unlink(partial.name)
            raise


def fetch_blob(
    images_dir: str,
    digest: str,
    open_blob: Callable[[], urllib.request.Request],
    timeout: float,
    stop: threading.Event,
) -> BinaryIO:
    """Open a cached layer, downloading it once under a cross-process lock.

    Open the file before releasing the lock so it remains readable even if
    another pull evicts its directory entry before this image applies it.
    """
    path = blob_path(images_dir, digest)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}.lock", "a", encoding="utf-8") as lock:
        while True:
            if stop.is_set():
                raise concurrent.futures.CancelledError()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                stop.wait(0.05)
        if os.path.isfile(path):
            os.utime(path)
        else:
            _download(open_blob, digest, path, timeout, stop)
            logger.info("Cached sandbox image layer %s", digest)
        return open(path, "rb")


def _close_blob(download: "concurrent.futures.Future[BinaryIO]") -> None:
    if not download.cancelled() and download.exception() is None:
        download.result().close()


@contextlib.contextmanager
def fetch_blobs(
    images_dir: str,
    digests: List[str],
    open_blob: Callable[[str], urllib.request.Request],
    timeout: float,
) -> Iterator[List[BinaryIO]]:
    """Fetch unique layers concurrently and expose them in manifest order."""
    raw = os.environ.get(_PULL_PARALLELISM_ENV, "4")
    try:
        workers = max(int(raw), 1)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: expected an integer.", _PULL_PARALLELISM_ENV, raw
        )
        workers = 4
    stop = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="sandbox-layer-download"
    )
    downloads = {
        digest: pool.submit(
            fetch_blob, images_dir, digest, lambda d=digest: open_blob(d), timeout, stop
        )
        for digest in dict.fromkeys(digests)
    }
    try:
        blobs = {digest: future.result() for digest, future in downloads.items()}
        yield [blobs[digest] for digest in digests]
    finally:
        stop.set()
        for download in downloads.values():
            download.cancel()
            download.add_done_callback(_close_blob)
        pool.shutdown(wait=False)


def evict_least_recently_used_blobs(
    images_dir: str, max_bytes: int, keep: Optional[List[str]] = None
) -> None:
    """Evict oldest layers past the cap, skipping kept or locked blobs."""
    if max_bytes <= 0:
        return
    directory = _blob_dir(images_dir)
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return
    kept = {blob_path(images_dir, digest) for digest in keep or []}
    entries = []
    now = time.time()
    for name in names:
        path = os.path.join(directory, name)
        if name.startswith(".partial-"):
            try:
                if now - os.stat(path).st_mtime > 3600:
                    os.unlink(path)
            except FileNotFoundError:
                pass
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", name):
            continue
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    total = sum(size for _, size, _ in entries)
    for mtime, size, path in sorted(entries):
        if total <= max_bytes:
            return
        if path in kept:
            continue
        try:
            with open(f"{path}.lock", "a", encoding="utf-8") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # A pull may have reused or replaced the blob since the scan.
                if os.stat(path).st_mtime != mtime:
                    continue
                os.unlink(path)
        except (BlockingIOError, FileNotFoundError):
            continue
        total -= size
        logger.info("Evicted cached sandbox image layer %s (%d bytes)", path, size)
