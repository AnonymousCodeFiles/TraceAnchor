from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Iterator, Mapping, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq


T = TypeVar("T")


def seconds_to_ns(value: object) -> int:
    return int(
        (Decimal(str(value)) * Decimal(1_000_000_000)).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_uid(relative_directory: str) -> str:
    normalized = Path(relative_directory).as_posix().strip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def scenario_token(uid: str, seed: int) -> str:
    material = f"traceanchor-agent:{seed}:{uid}".encode("ascii")
    return "tw_" + hashlib.sha256(material).hexdigest()[:24]


def batches(values: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def atomic_write_parquet(
    path: Path,
    schema: pa.Schema,
    rows: Iterable[Mapping[str, object]],
    batch_size: int = 4096,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    count = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(temp_name, schema=schema, compression="zstd")
        for batch in batches(rows, batch_size):
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(table)
            count += len(batch)
        writer.close()
        writer = None
        with open(temp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        return count
    except BaseException:
        if writer is not None:
            writer.close()
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temp_name, compression="zstd")
        with open(temp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
