import unittest

from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import make_assertion


class TestFunctionalApi(unittest.TestCase):
    def test_assertion(self):
        # discrete item assertions
        v = '1234'
        ia = make_assertion(state_element=v, negated=True)
        self.assertIsInstance(ia, ItemAssertion)
        self.assertIsInstance(ia.item, SymbolicItem)
        self.assertEqual(v, ia.item.state_element)
        self.assertEqual(True, ia.negated)
