from unittest import TestCase

from schema_mechanism.data_structures import ExtendedItemCollection
from schema_mechanism.data_structures import SchemaStats
from schema_mechanism.func_api import make_assertion
from test_share.test_classes import MockObserver


class TestExtendedItemCollection(TestCase):
    def setUp(self) -> None:
        self.eic = ExtendedItemCollection(schema_stats=SchemaStats())

    def test_relevant_items(self):
        self.assertEqual(0, len(self.eic.relevant_items))

        # verify immutability
        self.assertRaises(AttributeError, lambda: self.eic.relevant_items.add('x'))
        self.assertRaises(AttributeError, lambda: self.eic.relevant_items.clear())

    def test_update_relevant_items(self):
        self.assertEqual(0, len(self.eic.relevant_items))

        self.eic.update_relevant_items(make_assertion(1, negated=True))
        self.assertEqual(1, len(self.eic.relevant_items))

    def test_known_relevant_item(self):
        a1 = make_assertion(1)
        self.eic.update_relevant_items(a1)
        self.assertTrue(self.eic.known_relevant_item(a1))

        a2 = make_assertion(2)
        self.assertFalse(self.eic.known_relevant_item(a2))

    def test_register(self):
        obs = MockObserver()

        self.eic.register(obs)
        self.assertIn(obs, self.eic.observers)

    def test_notify_all(self):
        obs = MockObserver()

        self.eic.register(obs)

        self.eic.notify_all('one', 'two', three='three')

        # test that *args and **kwargs were successfully passed to observer
        self.assertIn('one', obs.last_message['args'])
        self.assertIn('two', obs.last_message['args'])
        self.assertIn('three', obs.last_message['kwargs'])

        # test that 'source' and 'n' were added to payload
        self.assertIn('source', obs.last_message['kwargs'])
        self.assertIs(self.eic, obs.last_message['kwargs']['source'])

        self.assertIn('n', obs.last_message['kwargs'])
        self.assertEqual(0, obs.last_message['kwargs']['n'])

        # test alternate 'source' is preserved
        self.eic.notify_all(source='alternate')
        self.assertIs('alternate', obs.last_message['kwargs']['source'])

        # test alternate 'n' is overwritten
        self.eic.notify_all(n='remove')
        self.assertIs(0, obs.last_message['kwargs']['n'])

        # test new relevant item increases 'n' to 1 in notify message, and is removed after notify_all
        a1 = make_assertion(1)
        self.eic.update_relevant_items(a1)
        self.assertTrue(self.eic.new_relevant_items())

        self.eic.notify_all()
        self.assertIs(1, obs.last_message['kwargs']['n'])

        # test that notify_all clears new relevant item state
        self.assertFalse(self.eic.new_relevant_items())

        # test that relevant items are preserved
        self.assertEqual(1, len(self.eic.relevant_items))

    def test_new_relevant_items(self):
        self.assertFalse(self.eic.new_relevant_items())

        a1 = make_assertion(1)
        self.eic.update_relevant_items(a1)

        self.assertTrue(self.eic.new_relevant_items())
