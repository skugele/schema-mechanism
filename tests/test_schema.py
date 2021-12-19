from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import Schema, Action, Context, Result, DiscreteItem, ContinuousItem, State


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

        # Verify that initial values are set properly
        self.assertEqual(Schema.INITIAL_RELIABILITY, s.reliability)

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
        c = Context([
            DiscreteItem('1'),
            DiscreteItem('2', negated=True),
            ContinuousItem(np.array([1.0, 0.0])),
            ContinuousItem(np.array([0.0, 1.0]), negated=True)
        ])

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be satisfied
        ##########################
        state = State(discrete_values=['1'],
                      continuous_values=[np.array([1.0, 0.0])])

        self.assertTrue(schema.context.is_satisfied(state))

        # expected to NOT be satisfied
        ##############################

        # case 1: present negated discrete item
        state = State(discrete_values=['1', '2'],
                      continuous_values=[np.array([1.0, 0.0])])

        self.assertFalse(schema.context.is_satisfied(state))

        # case 2: present negated continuous item
        state = State(discrete_values=['1'],
                      continuous_values=[np.array([1.0, 0.0]),
                                         np.array([0.0, 1.0])])

        self.assertFalse(schema.context.is_satisfied(state))

        # case 3: missing discrete element
        state = State(discrete_values=['3'],
                      continuous_values=[np.array([1.0, 0.0])])

        self.assertFalse(schema.context.is_satisfied(state))

        # case 4: missing continuous element
        state = State(discrete_values=['1'],
                      continuous_values=[np.array([0.5, 0.5])])

        self.assertFalse(schema.context.is_satisfied(state))
