import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SAVE_FILE_FORMAT = '{prefix}-{object_name}-v{version}.{suffix}'


def get_serialization_filename(object_name: str,
                               prefix: Optional[str] = None,
                               suffix: Optional[str] = None,
                               version: Optional[str] = None,
                               format_string: Optional[str] = None,
                               **kwargs) -> str:
    prefix = prefix or 'schema_mechanism'
    suffix = suffix or 'sav'
    version = version or '0.0.0'
    format_string = DEFAULT_SAVE_FILE_FORMAT if format_string is None else format_string

    return format_string.format(object_name=object_name,
                                prefix=prefix,
                                suffix=suffix,
                                version=version,
                                **kwargs)
