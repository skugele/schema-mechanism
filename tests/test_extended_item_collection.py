from unittest import TestCase

from schema_mechanism.core import ExtendedItemCollection
from schema_mechanism.core import SchemaSpinOffType
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_items
from schema_mechanism.util import repr_str
from test_share.test_classes import MockObserver
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestExtendedItemCollection(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.eic = ExtendedItemCollection(suppressed_items=sym_items('1;2;3'))

        self.obs = MockObserver()
        self.eic.register(self.obs)

    def test_suppress_list(self):
        self.assertSetEqual(self.eic.suppressed_items, set(sym_items('1;2;3')))

    # noinspection PyUnresolvedReferences
    def test_relevant_items(self):
        self.assertEqual(0, len(self.eic.relevant_items))

        # verify immutability
        self.assertRaises(AttributeError, lambda: self.eic.relevant_items.add('x'))
        self.assertRaises(AttributeError, lambda: self.eic.relevant_items.clear())

    def test_known_relevant_item(self):
        a1 = sym_item('1')
        self.eic.update_relevant_items(a1)
        self.assertTrue(a1 in self.eic.relevant_items)

        a2 = sym_item('2')
        self.assertFalse(a2 in self.eic.relevant_items)

    def test_register_and_unregister(self):
        obs = MockObserver()

        self.eic.register(obs)
        self.assertIn(obs, self.eic.observers)

        self.eic.unregister(obs)
        self.assertNotIn(obs, self.eic.observers)

    def test_update_relevant_items(self):
        # test: when suppressed == False, item argument should be added to the new relevant item set
        self.assertEqual(0, len(self.eic.new_relevant_items))
        self.eic.update_relevant_items(sym_item('1'), suppressed=False)
        self.assertEqual(1, len(self.eic.new_relevant_items))

    def test_update_relevant_items_with_suppressed(self):
        # test: when suppressed == True, relevant items should not be added to the new relevant item set
        self.assertEqual(0, len(self.eic.new_relevant_items))
        self.eic.update_relevant_items(sym_item('1'), suppressed=True)
        self.assertEqual(0, len(self.eic.new_relevant_items))

    def test_eq(self):
        self.assertTrue(satisfies_equality_checks(
            obj=self.eic,
            other_same_type=ExtendedItemCollection(suppressed_items=sym_items('4;5;6')),
            other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.eic))

    def test_str(self):
        values = '; '.join([f'{k} -> {v}' for k, v in self.eic.stats.items()])

        expected_str = f'{self.eic.__class__.__name__}[{values}]'
        self.assertEqual(expected_str, str(self.eic))

    def test_repr(self):
        self.eic.update_relevant_items(sym_item('1'), suppressed=False)

        item_stats = ', '.join(['{} -> {}'.format(k, v) for k, v in self.eic.stats.items()])
        relevant_items = ', '.join(str(i) for i in self.eic.relevant_items)
        new_relevant_items = ', '.join(str(i) for i in self.eic.new_relevant_items)

        attr_values = {
            'stats': f'[{item_stats}]',
            'relevant_items': f'[{relevant_items}]',
            'new_relevant_items': f'[{new_relevant_items}]',
        }

        expected_repr = repr_str(self.eic, attr_values)
        self.assertEqual(expected_repr, repr(self.eic))

    def test_new_relevant_items(self):
        # test: setter for new_relevant_items should properly set relevant items collection
        for item_collection in [list(), [sym_item('1')], {sym_item('2'), sym_item('3')}]:
            self.eic.new_relevant_items = item_collection
            self.assertSetEqual(set(item_collection), self.eic.new_relevant_items)

    def test_notify_all(self):
        self.eic.notify_all(spin_off_type=SchemaSpinOffType.CONTEXT, one='one', two='two', three='three')

        # test that *args and **kwargs were successfully passed to observer
        self.assertIn('one', self.obs.last_message['kwargs'])
        self.assertIn('two', self.obs.last_message['kwargs'])
        self.assertIn('three', self.obs.last_message['kwargs'])

        # test that 'source' was added to payload automatically
        self.assertIn('source', self.obs.last_message['kwargs'])
        self.assertIs(self.eic, self.obs.last_message['kwargs']['source'])

        # test notify_all clears new_relevant_items
        a1 = sym_item('1')
        self.eic.update_relevant_items(a1)
        self.assertEqual(1, len(self.eic.relevant_items))
        self.assertEqual(1, len(self.eic.new_relevant_items))

        self.eic.notify_all(spin_off_type=SchemaSpinOffType.RESULT)
        self.assertEqual(1, len(self.eic.relevant_items))
        self.assertEqual(0, len(self.eic.new_relevant_items))
