from pathlib import Path
from pickle import dump
from pickle import load
from typing import Any
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class Serializable(Protocol):
    def save(self, path: Path, overwrite: bool = False) -> None: ...

    def load(self, path: Path) -> None: ...


def check_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f'File {path} already exists. Use overwrite to replace.')


def check_readable(path: Path) -> None:
    if not (path.exists() and path.is_file()):
        raise ValueError(f'Unable to deserialize non-existent or unreadable file: {path}')


def serialize(obj: Any, path: Path, overwrite: bool = False) -> None:
    check_writable(path, overwrite)
    with path.open("wb") as file:
        dump(obj, file)


def deserialize(path: Path) -> Any:
    check_readable(path)
    with path.open("rb") as file:
        return load(file)
