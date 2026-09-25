"""Generic reader interface used by the canonical manifest builder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from open_badger.data.schema import DatasetMetadata, SamplePointer


class DatasetReader(Protocol):
    """Interface implemented by a backend-specific dataset reader."""

    def metadata(self) -> DatasetMetadata:
        """Return normalized dataset metadata."""

    def iter_pointers(self) -> Iterable[SamplePointer]:
        """Yield one pointer for every addressable frame."""

    def get_frame(self, pointer: SamplePointer) -> Mapping[str, Any]:
        """Resolve one pointer to a raw canonical frame."""
