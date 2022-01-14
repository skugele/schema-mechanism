from unittest import TestCase

from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.func_api import make_assertion


class TestStateAssertion(TestCase):
    def test_init(self):
        # Allow empty StateAssertion (no items)
        sa = StateAssertion()
        self.assertEqual(0, len(sa))

        # Support multiple assertions
        sa = StateAssertion((make_assertion('1'),
                             make_assertion('2', negated=True),
                             make_assertion('3')))

        self.assertEqual(3, len(sa))

    def test_is_satisfied(self):
        state = ['1', '2']

        # an empty StateAssertion should always be satisfied
        c = StateAssertion()
        self.assertTrue(c.is_satisfied(state))

        # single discrete item
        #######################

        # expected to be satisfied
        c = StateAssertion((make_assertion('1'),))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((make_assertion('3'),))
        self.assertFalse(c.is_satisfied(state))

        # multiple discrete items (all must be matched)
        ###############################################

        # expected to be satisfied
        c = StateAssertion((make_assertion('1'),
                            make_assertion('2')))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((make_assertion('1'),
                            make_assertion('3')))
        self.assertFalse(c.is_satisfied(state))

        c = StateAssertion((make_assertion('1'),
                            make_assertion('2'),
                            make_assertion('3')))
        self.assertFalse(c.is_satisfied(state))

    def test_is_contained(self):
        sa = StateAssertion((make_assertion('1'),
                             make_assertion('2')))

        # expected to be contained (discrete)
        self.assertTrue(make_assertion('1') in sa)

        # expected to be NOT contained (discrete)
        self.assertFalse(make_assertion('3') in sa)

    def test_replicate_with(self):
        sa = StateAssertion()

        # 1st discrete item should be added
        sa1 = sa.replicate_with(make_assertion('1'))
        self.assertIsNot(sa, sa1)
        self.assertTrue(make_assertion('1') in sa1)
        self.assertEqual(1, len(sa1))

        # 2nd discrete item should be added
        sa2 = sa1.replicate_with(make_assertion('2'))
        self.assertIsNot(sa1, sa2)
        self.assertTrue(make_assertion('2') in sa2)
        self.assertEqual(2, len(sa2))

        # identical discrete item should NOT be added
        try:
            sa2.replicate_with(make_assertion('2'))
            self.fail('Did\'t raise ValueError as expected!')
        except ValueError as e:
            self.assertEqual(str(e), 'ItemAssertion already exists in StateAssertion')
