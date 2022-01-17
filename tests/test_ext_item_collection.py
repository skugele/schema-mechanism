from unittest import TestCase

from schema_mechanism.data_structures import ExtendedItemCollection
from schema_mechanism.data_structures import SchemaStats
from schema_mechanism.func_api import make_assertion
from test_share.test_classes import MockObserver


class TestExtendedItemCollection(TestCase):
    def setUp(self) -> None:
        self.eic = ExtendedItemCollection(schema_stats=SchemaStats())

        self.obs = MockObserver()
        self.eic.register(self.obs)

    def test_relevant_items(self):
        self.assertEqual(0, len(self.eic.relevant_items))

        # verify immutability
        self.assertRaises(AttributeError, lambda: self.eic.relevant_items.add('x'))
        self.assertRaises(AttributeError, lambda: self.eic.relevant_items.clear())

    def test_known_relevant_item(self):
        a1 = make_assertion(1)
        self.eic.update_relevant_items(a1)
        self.assertTrue(self.eic.known_relevant_item(a1))

        a2 = make_assertion(2)
        self.assertFalse(self.eic.known_relevant_item(a2))

    def test_register_and_unregister(self):
        obs = MockObserver()

        self.eic.register(obs)
        self.assertIn(obs, self.eic.observers)

        self.eic.unregister(obs)
        self.assertNotIn(obs, self.eic.observers)

    def test_update_relevant_items(self):
        self.assertEqual(0, len(self.eic.new_relevant_items))
        self.eic.update_relevant_items(make_assertion(1))
        self.assertEqual(1, len(self.eic.new_relevant_items))

    def test_notify_all(self):
        self.eic.notify_all('one', 'two', three='three')

        # test that *args and **kwargs were successfully passed to observer
        self.assertIn('one', self.obs.last_message['args'])
        self.assertIn('two', self.obs.last_message['args'])
        self.assertIn('three', self.obs.last_message['kwargs'])

        # test that 'source' was added to payload automatically
        self.assertIn('source', self.obs.last_message['kwargs'])
        self.assertIs(self.eic, self.obs.last_message['kwargs']['source'])

        # test alternate 'source' is preserved
        self.eic.notify_all(source='alternate')
        self.assertIs('alternate', self.obs.last_message['kwargs']['source'])

        # test notify_all clears new_relevant_items
        a1 = make_assertion(1)
        self.eic.update_relevant_items(a1)
        self.assertEqual(1, len(self.eic.relevant_items))
        self.assertEqual(1, len(self.eic.new_relevant_items))

        self.eic.notify_all()
        self.assertEqual(1, len(self.eic.relevant_items))
        self.assertEqual(0, len(self.eic.new_relevant_items))
