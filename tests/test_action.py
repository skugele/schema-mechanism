from unittest import TestCase

from schema_mechanism.data_structures import Action


class TestAction(TestCase):
    def test_init(self):
        # verify that id field can be explicitly set
        a = Action('1')
        self.assertTrue('1', a.id)

        # verify that a default id is provided
        a = Action()
        self.assertIsNotNone(a.id)

        # verify that default ids are (likely) unique
        actions = [Action() for _ in range(100)]
        self.assertTrue(len(actions), len(set(actions)))
