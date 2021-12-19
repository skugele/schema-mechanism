from unittest import TestCase

from schema_mechanism.data_structures import Schema, Action, Context, Result, DiscreteItem


class TestSchema(TestCase):
    def test_init(self):
        # Action CANNOT be None
        try:
            Schema(action=None)
            self.fail('Action=None should generate a ValueError')
        except ValueError as e:
            self.assertEqual('Action cannot be None', str(e))

        # Context and Action CAN be None
        try:
            s = Schema(action=Action('My Action'))
            self.assertIsNone(s.context)
            self.assertIsNone(s.result)
        except Exception as e:
            self.fail(f'Unexpected exception raised: {e}')

        # Verify that id is assigned automatically
        s = Schema(Action())
        self.assertIsNotNone(s.id)

        # Verify immutability
        s = Schema(context=Context([DiscreteItem('1')]),
                   action=Action(),
                   result=Result([DiscreteItem('2')]))

        try:
            s.context = Context()
            self.fail('Schema\'s context is not immutable as expected')
        except Exception as e:
            pass

        try:
            s.result = Result()
            self.fail('Schema\'s result is not immutable as expected')
        except Exception as e:
            pass

        try:
            s.action = Action()
            self.fail('Schema\'s action is not immutable as expected')
        except Exception as e:
            pass

    def test_is_satisfied(self):
        self.fail('Not Implemented')
