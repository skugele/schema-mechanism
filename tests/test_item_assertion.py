from unittest import TestCase

from schema_mechanism.data_structures import DiscreteItem, ItemAssertion, State


class TestItemAssertion(TestCase):
    def test_init(self):
        i = DiscreteItem('1234')
        ia = ItemAssertion(item=i)
        self.assertEqual(i, ia.item)

        # check default negated value is False
        self.assertEqual(False, ia.negated)

        # check setting of non-default negated value to True
        ia = ItemAssertion(item=i, negated=True)
        self.assertEqual(True, ia.negated)

        # check immutability
        try:
            ia.negated = False
            self.fail('ItemAssertion is not immutable as expected: able to set negated directly')
        except Exception:
            pass

    def test_is_satisfied(self):
        # not negated
        #############
        i = DiscreteItem(value='1234')
        ia = ItemAssertion(item=i, negated=False)

        # item expected to be ON for these states
        self.assertTrue(ia.is_satisfied(state=State(discrete_values=['1234'])))
        self.assertTrue(ia.is_satisfied(state=State(discrete_values=['123', '1234'])))

        # item expected to be OFF for these states
        self.assertFalse(ia.is_satisfied(state=State()))
        self.assertFalse(ia.is_satisfied(state=State(discrete_values=['123'])))
        self.assertFalse(ia.is_satisfied(state=State(discrete_values=['123', '4321'])))

        # negated
        #########
        i = DiscreteItem(value='1234')
        ia = ItemAssertion(item=i, negated=True)

        # item expected to be ON for these states
        self.assertFalse(ia.is_satisfied(state=State(discrete_values=['1234'])))
        self.assertFalse(ia.is_satisfied(state=State(discrete_values=['123', '1234'])))

        # item expected to be OFF for these states
        self.assertTrue(ia.is_satisfied(state=State()))
        self.assertTrue(ia.is_satisfied(state=State(discrete_values=['123'])))
        self.assertTrue(ia.is_satisfied(state=State(discrete_values=['123', '4321'])))
