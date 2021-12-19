from unittest import TestCase

from schema_mechanism.data_structures import Action


class TestAction(TestCase):
    def test_init(self):
        # verify that id field can be explicitly set
        a = Action('1')
        self.assertTrue('1', a.uid)

        # verify that a default id is provided
        a = Action()
        self.assertIsNotNone(a.uid)

        # verify that default ids are (likely) unique
        actions = [Action() for _ in range(100)]
        self.assertTrue(len(actions), len(set(actions)))

    def test_equals(self):
        a1 = Action()
        a2 = Action()

        self.assertEqual(a1, a1)
        self.assertNotEqual(a1, a2)

        a3 = Action(uid='my action')
        a4 = Action(uid='my action')
        self.assertEqual(a3, a4)
