import logging
from pathlib import Path
from time import time
from typing import Any
from typing import Optional

from dill import dump
from dill import load

logger = logging.getLogger(__name__)

DEFAULT_SAVE_FILE_FORMAT = '{prefix}{object_name}-{version}-{unique_id}{suffix}'


def get_unique_id() -> str:
    return str(int(time()))


def get_serialization_filename(object_name: str,
                               prefix: Optional[str] = None,
                               suffix: Optional[str] = None,
                               version: Optional[str] = None,
                               format_string: Optional[str] = None) -> str:
    prefix = prefix or ''
    suffix = suffix or ''
    version = version or 'v0.0'
    unique_id = get_unique_id()
    format_string = DEFAULT_SAVE_FILE_FORMAT if format_string is None else format_string

    return format_string.format(object_name=object_name,
                                unique_id=unique_id,
                                version=version,
                                prefix=prefix,
                                suffix=suffix)


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
