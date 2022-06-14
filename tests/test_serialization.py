import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

import schema_mechanism.serialization.json.decoders
import schema_mechanism.serialization.json.encoders
import schema_mechanism.versioning
from schema_mechanism.core import Action
from schema_mechanism.func_api import sym_item
from schema_mechanism.serialization import deserialize
from schema_mechanism.serialization import get_serialization_filename
from schema_mechanism.serialization import serialize
from schema_mechanism.util import check_readable
from schema_mechanism.util import check_writable
from test_share.test_func import common_test_setup


class TestSerialization(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_check_writable(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-serialization-check_writable.sav'))

            # create file that would be written to
            path.touch()

            # test: SHOULD raise a ValueError when overwrite is False
            self.assertRaises(ValueError, lambda: check_writable(path=path, overwrite=False))

            # test: SHOULD NOT raise a ValueError when overwrite is True
            try:
                check_writable(path=path, overwrite=True)
            except ValueError as e:
                self.fail(f'Raised unexpected ValueError: {e}')

    def test_check_readable(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-serialization-check_readable.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            # test: SHOULD raise a ValueError when file does not exist
            self.assertRaises(ValueError, lambda: check_readable(path=path))

            # create file to be checked
            path.touch()

            # test: SHOULD NOT raise a ValueError when file exists and is readable
            try:
                check_readable(path=path)
            except ValueError as e:
                self.fail(f'Raised unexpected ValueError: {e}')

    def test_get_serialization_file_path(self):
        expected_object_name = 'ObjectName'
        expected_version = 'v1.0'
        expected_prefix = 'P'
        expected_suffix = 'S'
        expected_additional_element = 'A'

        filename = get_serialization_filename(
            object_name=expected_object_name,
            version=expected_version,
            prefix=expected_prefix,
            suffix=expected_suffix,
            additional_element=expected_additional_element,
            format_string='{prefix}-{object_name}-{version}-{additional_element}-{suffix}')

        prefix, object_name, version, additional_element, suffix = filename.split('-')

        self.assertEqual(expected_prefix, prefix)
        self.assertEqual(expected_object_name, object_name)
        self.assertEqual(expected_version, version)
        self.assertEqual(expected_additional_element, additional_element)
        self.assertEqual(expected_suffix, suffix)

        # test: check that defaults should produce expected results
        default_prefix = 'schema_mechanism'
        default_suffix = schema_mechanism.serialization.DEFAULT_ENCODING
        default_version = schema_mechanism.versioning.version

        filename = get_serialization_filename(object_name=object_name)

        self.assertEqual(f'{default_prefix}-{object_name}-v{default_version}.{default_suffix}', filename)

    def test_serialize_and_deserialize(self):
        # try serializing/deserializing an object containing both built-in and custom types
        original = [101, '101', None, [], {'one': 1, 'two': 2}, sym_item('1'), [Action('test action')]]

        encoder = schema_mechanism.serialization.json.encoders.encode
        decoder = schema_mechanism.serialization.json.decoders.decode

        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-serialization-serialize_and_deserialize.sav'))

            # sanity check: file SHOULD NOT exist before serialize
            self.assertFalse(path.exists())

            # test: deserialize should raise a ValueError if file does not exist
            self.assertRaises(ValueError, lambda: deserialize(decoder=decoder, path=path))

            object_registry: dict[int, Any] = dict()

            serialize(
                original,
                encoder=encoder,
                path=path,
                overwrite=False,
                object_registry=object_registry
            )

            # sanity check: file SHOULD exist after serialize
            self.assertTrue(path.exists())

            restored = deserialize(
                path=path,
                decoder=decoder,
                object_registry=object_registry
            )

            # test: the deserialized object should equal the original
            self.assertListEqual(original, restored)

            # test: a ValueError SHOULD be raised when trying to overwrite file if overwrite is False
            self.assertRaises(ValueError, lambda: serialize(original, encoder=encoder, path=path, overwrite=False))

            # test: a ValueError SHOULD NOT be raised when trying to overwrite file if overwrite is True
            try:
                serialize(original, encoder=encoder, path=path, overwrite=True)
            except ValueError as e:
                self.fail(f'Unexpected ValueError was raised: {e}')
