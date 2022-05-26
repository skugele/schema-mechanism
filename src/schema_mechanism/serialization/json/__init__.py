from pathlib import Path
from typing import Any
from typing import Callable

import schema_mechanism.serialization.json.decoders as json_decoders
import schema_mechanism.serialization.json.encoders as json_encoders
from schema_mechanism.util import check_readable
from schema_mechanism.util import check_writable


def serialize(obj: Any, /, path: Path, encoder: Callable = None, overwrite: bool = False) -> None:
    check_writable(path, overwrite)

    encoder = encoder if encoder else json_encoders.encode
    encoded_obj = encoder(obj)

    with path.open("w") as fp:
        fp.write(encoded_obj)


def deserialize(path: Path, decoder: Callable = None) -> Any:
    check_readable(path)

    with path.open("r") as fp:
        encoded_obj = fp.read()

        decoder = decoder if decoder else json_decoders.decode
        obj = decoder(encoded_obj)

        return obj
