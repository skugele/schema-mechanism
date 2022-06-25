import re
from typing import Optional
from unittest import TestCase

from examples.optimizers import get_optimizer_report_filename


class TestSharedOptimizerFunctions(TestCase):

    def test_get_optimizer_report_filename(self):
        prefix = 'prefix'
        study_name = 'study_name'
        sampler = 'sampler'
        pruner = 'pruner'

        test_conditions = [
            {'prefix': prefix, 'study_name': study_name, 'sampler': sampler, 'pruner': pruner},
            {'prefix': prefix, 'study_name': None, 'sampler': sampler, 'pruner': pruner},
            {'prefix': prefix, 'study_name': study_name, 'sampler': None, 'pruner': pruner},
            {'prefix': prefix, 'study_name': study_name, 'sampler': sampler, 'pruner': None},
            {'prefix': prefix, 'study_name': None, 'sampler': sampler, 'pruner': pruner},
            {'prefix': prefix, 'study_name': None, 'sampler': sampler, 'pruner': None},
            {'prefix': prefix, 'study_name': study_name, 'sampler': None, 'pruner': None},
            {'prefix': prefix, 'study_name': None, 'sampler': None, 'pruner': pruner},
            {'prefix': prefix, 'study_name': None, 'sampler': None, 'pruner': None},
        ]

        for kwargs in test_conditions:
            # test case: check without timestamp
            filename = get_optimizer_report_filename(add_timestamp=False, **kwargs)
            self._check_optimizer_report_filename(filename=filename, with_timestamp=False, **kwargs)

            # test case: check with timestamp
            filename = get_optimizer_report_filename(add_timestamp=True, **kwargs)
            self._check_optimizer_report_filename(filename=filename, with_timestamp=True, **kwargs)

    def _check_optimizer_report_filename(self,
                                         filename: str,
                                         prefix: Optional[str],
                                         study_name: Optional[str],
                                         sampler: Optional[str],
                                         pruner: Optional[str],
                                         with_timestamp: bool) -> None:
        report_name, file_extension = filename.split('.')

        # test: filename should have a file extension of csv
        self.assertEqual(file_extension, 'csv')

        components: list[str] = report_name.split('-')

        # remove timestamp
        if with_timestamp and len(components) >= 1:
            timestamp = components.pop()

            # test: if timestamp added, it should be represented as a string of integers
            if not re.match(r'^\d+$', string=timestamp):
                self.fail('Invalid or missing timestamp')

        expected_components = [
            component for component in
            [prefix, study_name, sampler, pruner]
            if component
        ]

        # test: all non-empty, non-none components should match expected values in the correct order
        self.assertListEqual(expected_components, components)
