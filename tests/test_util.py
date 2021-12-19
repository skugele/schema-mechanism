from unittest import TestCase

from schema_mechanism.util import get_unique_id


class Test(TestCase):
    def test_get_unique_id(self):
        ids = []
        for _ in range(1000):
            uid = get_unique_id()
            self.assertNotIn(uid, ids)

            ids.append(uid)
