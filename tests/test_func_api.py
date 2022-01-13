import unittest

from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import make_assertion
from schema_mechanism.func_api import make_assertions


class TestFunctionalApi(unittest.TestCase):
    def test_make_assertion(self):
        # discrete item assertions
        v = '1234'
        ia = make_assertion(state_element=v, negated=True)
        self.assertIsInstance(ia, ItemAssertion)
        self.assertIsInstance(ia.item, SymbolicItem)
        self.assertEqual(v, ia.item.state_element)
        self.assertEqual(True, ia.negated)

    def test_make_assertions(self):
        n_items = 1000
        ias = make_assertions(range(n_items))
        self.assertEqual(n_items, len(ias))

        for ia in ias:
            # verify item types are ItemAssertion
            self.assertIsInstance(ia, ItemAssertion)

            # verify negated value (False) applied to all item assertions
            self.assertFalse(ia.negated)

        # verify all state elements in returned collection
        self.assertTrue(all({make_assertion(se) in ias for se in range(n_items)}))

        ias = make_assertions(range(n_items), negated=True)
        self.assertEqual(n_items, len(ias))

        for ia in ias:
            # verify item types are ItemAssertion
            self.assertIsInstance(ia, ItemAssertion)

            # verify negated value (True) applied to all item assertions
            self.assertTrue(ia.negated)

        # verify all state elements in returned collection
        self.assertTrue(all({make_assertion(se, negated=True) in ias for se in range(n_items)}))
