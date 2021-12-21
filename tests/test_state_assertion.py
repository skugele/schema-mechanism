from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import StateAssertion, DiscreteItem, ContinuousItem, State


class TestStateAssertion(TestCase):
    def test_init(self):
        # Allow empty StateAssertion (no items)
        sa = StateAssertion()
        self.assertEqual(0, len(sa))

        # Allow mixture of discrete and continuous items
        sa = StateAssertion((
            DiscreteItem('1'),
            ContinuousItem(np.array([1.0, 2.0]))))

        self.assertEqual(2, len(sa))

    def test_immutability(self):
        sa = StateAssertion((
            DiscreteItem('1'),
            ContinuousItem(np.array([1.0, 0.0]))))

        # Try changing individual item
        try:
            sa.items[0] = DiscreteItem('5')
            self.fail('StateAssertion\'s items are not immutable as expected!')
        except TypeError as e:
            pass

        # Try changing the tuple referenced by items
        try:
            sa.items = (DiscreteItem('5'),)
            self.fail('StateAssertion\'s items are not immutable as expected!')
        except AttributeError as e:
            print(type(e))

    def test_is_satisfied(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])

        state = State(discrete_values=['1', '2'],
                      continuous_values=[v1, v2])

        # an empty StateAssertion should always be satisfied
        c = StateAssertion()
        self.assertTrue(c.is_satisfied(state))

        # single discrete item
        #######################

        # expected to be satisfied
        c = StateAssertion((DiscreteItem('1'),))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((DiscreteItem('3'),))
        self.assertFalse(c.is_satisfied(state))

        # multiple discrete items (all must be matched)
        ###############################################

        # expected to be satisfied
        c = StateAssertion((DiscreteItem('1'), DiscreteItem('2')))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((DiscreteItem('1'), DiscreteItem('3')))
        self.assertFalse(c.is_satisfied(state))

        c = StateAssertion((DiscreteItem('1'), DiscreteItem('2'), DiscreteItem('3')))
        self.assertFalse(c.is_satisfied(state))

        # single continuous item
        ########################

        # expected to be satisfied
        c = StateAssertion((ContinuousItem(v1),))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((ContinuousItem(v3),))
        self.assertFalse(c.is_satisfied(state))

        # multiple continuous items (all must be matched)
        #################################################

        # expected to be satisfied
        c = StateAssertion((ContinuousItem(v1), ContinuousItem(v2)))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((ContinuousItem(v1), ContinuousItem(v3)))
        self.assertFalse(c.is_satisfied(state))

        c = StateAssertion((ContinuousItem(v1), ContinuousItem(v2), ContinuousItem(v3)))
        self.assertFalse(c.is_satisfied(state))

        # mixed discrete and continuous
        ###############################

        # expected to be satisfied
        c = StateAssertion((DiscreteItem('1'), ContinuousItem(v1)))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((DiscreteItem('1'), ContinuousItem(v1), ContinuousItem(v3)))
        self.assertFalse(c.is_satisfied(state))

    def test_is_contained(self):
        sa = StateAssertion(items=(
            DiscreteItem('1'),
            DiscreteItem('2'),
            ContinuousItem(np.array([1.0, 0.0, 0.0])),
            ContinuousItem(np.array([0.0, 1.0, 0.0]))
        ))

        # expected to be contained (discrete)
        self.assertTrue(DiscreteItem('1') in sa)

        # expected to be contained (continuous)
        self.assertTrue(ContinuousItem(np.array([0.0, 1.0, 0.0])) in sa)

        # expected to be NOT contained (discrete)
        self.assertFalse(DiscreteItem('3') in sa)

        # expected to be NOT contained (continuous)
        self.assertFalse(ContinuousItem(np.array([0.0, 0.0, 1.0])) in sa)

    def test_replicate_with(self):
        sa = StateAssertion()

        # 1st discrete item should be added
        sa1 = sa.replicate_with(DiscreteItem('1'))
        self.assertIsNot(sa, sa1)
        self.assertTrue(DiscreteItem('1') in sa1)
        self.assertEqual(1, len(sa1))

        # 2nd discrete item should be added
        sa2 = sa1.replicate_with(DiscreteItem('2'))
        self.assertIsNot(sa1, sa2)
        self.assertTrue(DiscreteItem('2') in sa2)
        self.assertEqual(2, len(sa2))

        # identical discrete item should NOT be added
        try:
            sa2.replicate_with(DiscreteItem('2'))
            self.fail('Did\'t raise ValueError as expected!')
        except ValueError as e:
            self.assertEqual(str(e), 'Item already exists in StateAssertion')

        # 1st continuous item should be added
        sa3 = sa2.replicate_with(ContinuousItem(np.array([1.0, 0.0, 0.0])))
        self.assertIsNot(sa2, sa3)
        self.assertTrue(ContinuousItem(np.array([1.0, 0.0, 0.0])) in sa3)
        self.assertEqual(3, len(sa3))

        # 2nd continuous item should be added
        sa4 = sa3.replicate_with(ContinuousItem(np.array([0.0, 1.0, 0.0])))
        self.assertIsNot(sa3, sa4)
        self.assertTrue(ContinuousItem(np.array([0.0, 1.0, 0.0])) in sa4)
        self.assertEqual(4, len(sa4))

        # identical continuous item should NOT be added
        try:
            sa4.replicate_with(ContinuousItem(np.array([0.0, 1.0, 0.0])))
            self.fail('Did\'t raise ValueError as expected!')
        except ValueError as e:
            self.assertEqual(str(e), 'Item already exists in StateAssertion')
