from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import Context, DiscreteItem, ContinuousItem, State


class TestContext(TestCase):
    def test_init(self):
        # Allow empty context (no items)
        c = Context()
        self.assertEqual(0, len(c))

        # Allow mixture of discrete and continuous items
        c1 = Context(
            [
                DiscreteItem('1'),
                ContinuousItem(np.array([1.0, 2.0]))
            ])

        self.assertEqual(2, len(c1))

    def test_is_satisfied(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])

        state = State(discrete_values=['1', '2'],
                      continuous_values=[v1, v2])

        # an empty context should always be satisfied
        c = Context()
        self.assertTrue(c.is_satisfied(state))

        # single discrete item
        #######################

        # expected to be satisfied
        c = Context([DiscreteItem('1')])
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = Context([DiscreteItem('3')])
        self.assertFalse(c.is_satisfied(state))

        # multiple discrete items (all must be matched)
        ###############################################

        # expected to be satisfied
        c = Context([DiscreteItem('1'), DiscreteItem('2')])
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = Context([DiscreteItem('1'), DiscreteItem('3')])
        self.assertFalse(c.is_satisfied(state))

        c = Context([DiscreteItem('1'), DiscreteItem('2'), DiscreteItem('3')])
        self.assertFalse(c.is_satisfied(state))

        # single continuous item
        ########################

        # expected to be satisfied
        c = Context([ContinuousItem(v1)])
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = Context([ContinuousItem(v3)])
        self.assertFalse(c.is_satisfied(state))

        # multiple continuous items (all must be matched)
        #################################################

        # expected to be satisfied
        c = Context([ContinuousItem(v1), ContinuousItem(v2)])
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = Context([ContinuousItem(v1), ContinuousItem(v3)])
        self.assertFalse(c.is_satisfied(state))

        c = Context([ContinuousItem(v1), ContinuousItem(v2), ContinuousItem(v3)])
        self.assertFalse(c.is_satisfied(state))

        # mixed discrete and continuous
        ###############################

        # expected to be satisfied
        c = Context([DiscreteItem('1'), ContinuousItem(v1)])
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = Context([DiscreteItem('1'), ContinuousItem(v1), ContinuousItem(v3)])
        self.assertFalse(c.is_satisfied(state))
