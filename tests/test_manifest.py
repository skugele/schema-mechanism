import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from schema_mechanism.serialization import DEFAULT_ENCODING
from schema_mechanism.serialization import create_manifest
from schema_mechanism.serialization import get_manifest_filename
from schema_mechanism.serialization import load_manifest
from schema_mechanism.serialization import save_manifest
from schema_mechanism.versioning import version


class TestManifest(TestCase):

    def test_create_manifest(self):
        before_time = datetime.datetime.now().astimezone()

        manifest = create_manifest()

        # test: version element should be present and equal the current software version
        self.assertEqual(manifest['version'], version)

        # test: encoding element should be present and equal to default when another encoding is not provided
        self.assertEqual(manifest['encoding'], DEFAULT_ENCODING)

        # test: creation time element should be present
        self.assertIsInstance(manifest['creation_time'], datetime.datetime)
        self.assertGreaterEqual(manifest['creation_time'], before_time)

        # test: object_registry element should be added as None
        self.assertEqual(manifest['object_registry'], None)

        # test: objects element should have been added as an empty dict
        self.assertDictEqual(dict(), manifest['objects'])

        alternate_encoding = 'alternate'
        manifest = create_manifest(encoding=alternate_encoding)

        # test: encoding element should be present and equal to given value
        self.assertEqual(manifest['encoding'], alternate_encoding)

    def test_save_and_load_manifest(self):
        manifest = create_manifest()

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            manifest_filepath = path / get_manifest_filename(object_name='manifest')

            # test: file should not exist before save_manifest is called
            self.assertFalse(manifest_filepath.exists())

            save_manifest(manifest, path=path)

            # test: file should exist after save_manifest is called
            self.assertTrue(manifest_filepath.exists())

            # test: loaded manifest should equal the original manifest
            loaded_manifest = load_manifest(path=path)
            self.assertDictEqual(manifest, loaded_manifest)

            # test: save_manifest should raise a ValueError is file already exists and overwrite=False
            self.assertRaises(ValueError, lambda: save_manifest(manifest, path=path))

            try:
                new_manifest = create_manifest(encoding='alternate encoding')

                last_modification_time = manifest_filepath.stat().st_mtime_ns
                save_manifest(new_manifest, path=path, overwrite=True)
                new_modification_time = manifest_filepath.stat().st_mtime_ns

                self.assertGreater(new_modification_time, last_modification_time)

                loaded_manifest = load_manifest(path=path)
                self.assertDictEqual(new_manifest, loaded_manifest)
            except ValueError as e:
                self.fail(f'Unexpected ValueError: {e}')
