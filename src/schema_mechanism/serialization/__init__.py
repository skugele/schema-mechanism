import datetime
import logging
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional

import yaml

import schema_mechanism.serialization.json.decoders as json_decoders
import schema_mechanism.serialization.json.encoders as json_encoders
import schema_mechanism.versioning
from schema_mechanism.util import check_readable
from schema_mechanism.util import check_writable

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = 'json'
DEFAULT_SAVE_FILE_FORMAT = '{prefix}-{object_name}-v{version}.{suffix}'


def get_serialization_filename(object_name: str,
                               prefix: Optional[str] = None,
                               suffix: Optional[str] = None,
                               version: Optional[str] = None,
                               format_string: Optional[str] = None,
                               **kwargs) -> str:
    prefix = prefix or 'schema_mechanism'
    suffix = suffix or DEFAULT_ENCODING
    version = version or schema_mechanism.versioning.version
    format_string = DEFAULT_SAVE_FILE_FORMAT if format_string is None else format_string

    return format_string.format(
        object_name=object_name,
        prefix=prefix,
        suffix=suffix,
        version=version,
        **kwargs
    )


def serialize(obj: Any, /, path: Path, encoder: Callable, overwrite: bool = False, **kwargs) -> None:
    check_writable(path, overwrite)

    encoded_obj = encoder(obj, **kwargs)

    with path.open("w") as fp:
        fp.write(encoded_obj)


def deserialize(path: Path, decoder: Callable, **kwargs) -> Any:
    check_readable(path)

    with path.open("r") as fp:
        encoded_obj = fp.read()

        obj = decoder(encoded_obj, **kwargs)

        return obj


# type alias for manifest
Manifest = dict

DEFAULT_MANIFEST_FILE_FORMAT = '{prefix}-manifest-v{version}.yaml'


def get_manifest_filename(
        prefix: Optional[str] = None,
        version: Optional[str] = None,
        format_string: Optional[str] = None,
        **kwargs) -> str:
    prefix = prefix or 'schema_mechanism'
    version = version or schema_mechanism.versioning.version
    format_string = DEFAULT_MANIFEST_FILE_FORMAT if format_string is None else format_string

    return format_string.format(
        prefix=prefix,
        version=version,
        **kwargs
    )


def create_manifest(encoding: str = None) -> Manifest:
    manifest: Manifest = dict()

    manifest['version'] = schema_mechanism.versioning.version
    manifest['creation_time'] = datetime.datetime.now().astimezone()
    manifest['encoding'] = encoding or DEFAULT_ENCODING
    manifest['object_registry'] = None
    manifest['objects'] = dict()

    return manifest


def save_manifest(manifest: Manifest, /, path: Path, overwrite: bool = False) -> None:
    manifest_filepath: Path = path / get_manifest_filename()
    if manifest_filepath.exists() and not overwrite:
        raise ValueError(f'{manifest_filepath} already exists! Set overwrite to True to replace this file.')

    with manifest_filepath.open(mode='w') as fp:
        yaml.dump(stream=fp, data=manifest, Dumper=yaml.Dumper, sort_keys=False)


def load_manifest(path: Path) -> Manifest:
    manifest_filepath: Path = path / get_manifest_filename()
    with manifest_filepath.open(mode='r') as fp:
        manifest = yaml.load(stream=fp, Loader=yaml.Loader)
        return manifest


# type alias for object registry
ObjectRegistry = dict[str, Any]


def create_object_registry() -> ObjectRegistry:
    return dict()


def save_object_registry(
        object_registry: ObjectRegistry,
        /,
        manifest: Manifest,
        path: Path,
        encoder: Callable = None,
        overwrite: bool = False) -> None:
    if not object_registry:
        return

    object_registry_filepath: Path = path / get_serialization_filename(object_name='object_registry')
    serialize(
        object_registry,
        encoder=encoder,
        path=object_registry_filepath,
        overwrite=overwrite,
    )
    manifest['object_registry'] = str(object_registry_filepath)


def load_object_registry(path: Path, decoder: Callable = None) -> ObjectRegistry:
    return deserialize(path, decoder=decoder)


encoder_map = {
    DEFAULT_ENCODING: json_encoders.encode
}

decoder_map = {
    DEFAULT_ENCODING: json_decoders.decode
}
