import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from schema_mechanism.persistence import check_readable
from schema_mechanism.persistence import check_writable
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import get_serialization_filename
from schema_mechanism.persistence import serialize
from test_share.test_func import common_test_setup


class TestPersistence(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_check_writable(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-persistence-check_writable.sav'))

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
            path = Path(os.path.join(tmp_dir, 'test-file-persistence-check_readable.sav'))

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

        filename = get_serialization_filename(object_name=expected_object_name,
                                              version=expected_version,
                                              prefix=expected_prefix,
                                              suffix=expected_suffix,
                                              format_string='{prefix}-{object_name}-{version}-{unique_id}-{suffix}')

        prefix, object_name, version, unique_id, suffix = filename.split('-')

        self.assertEqual(expected_prefix, prefix)
        self.assertEqual(expected_object_name, object_name)
        self.assertEqual(expected_version, version)
        self.assertGreater(len(unique_id), 0)
        self.assertEqual(expected_suffix, suffix)

    def test_serialize_and_deserialize(self):
        original = [101, '101', None, [], {1, 2, 3}, {'one': 1, 'two': 2}]

        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-persistence-serialize_and_deserialize.sav'))

            # sanity check: file SHOULD NOT exist before serialize
            self.assertFalse(path.exists())

            # test: deserialize should raise a ValueError if file does not exist
            self.assertRaises(ValueError, lambda: deserialize(path=path))

            serialize(obj=original, path=path, overwrite=False)

            # sanity check: file SHOULD exist after serialize
            self.assertTrue(path.exists())

            # test: the deserialized object should equal the original
            self.assertListEqual(original, deserialize(path))

            # test: a ValueError SHOULD be raised when trying to overwrite file if overwrite is False
            self.assertRaises(ValueError, lambda: serialize(obj=original, path=path, overwrite=False))

            # test: a ValueError SHOULD NOT be raised when trying to overwrite file if overwrite is True
            try:
                serialize(obj=original, path=path, overwrite=True)
            except ValueError as e:
                self.fail(f'Unexpected ValueError was raised: {e}')
