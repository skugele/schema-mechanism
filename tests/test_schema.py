from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Context
from schema_mechanism.data_structures import ContinuousItem
from schema_mechanism.data_structures import DiscreteItem
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import Result
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateAssertion


class TestSchema(TestCase):
    def test_init(self):
        # Action CANNOT be None
        try:
            Schema(action=None)
            self.fail('Action=None should generate a ValueError')
        except ValueError as e:
            self.assertEqual('Action cannot be None', str(e))

        # Context and Result CAN be None
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
        s = Schema(context=Context(items=(ItemAssertion(DiscreteItem('1')),)),
                   action=Action(),
                   result=Result(items=(ItemAssertion(DiscreteItem('2')),)))

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

    def test_is_context_satisfied(self):
        c = Context((
            ItemAssertion(DiscreteItem('1')),
            ItemAssertion(DiscreteItem('2'), negated=True),
            ItemAssertion(ContinuousItem(np.array([1.0, 0.0]))),
            ItemAssertion(ContinuousItem(np.array([0.0, 1.0])), negated=True)
        ))

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be satisfied
        ##########################

        self.assertTrue(schema.context.is_satisfied(state=['1', np.array([1.0, 0.0])]))

        # expected to NOT be satisfied
        ##############################

        # case 1: present negated discrete item
        self.assertFalse(schema.context.is_satisfied(state=['1', '2', np.array([1.0, 0.0])]))

        # case 2: present negated continuous item
        self.assertFalse(schema.context.is_satisfied(state=['1', np.array([1.0, 0.0]), np.array([0.0, 1.0])]))

        # case 3: missing discrete element
        self.assertFalse(schema.context.is_satisfied(state=['3', np.array([1.0, 0.0])]))

        # case 4: missing continuous element
        self.assertFalse(schema.context.is_satisfied(state=['1', np.array([0.5, 0.5])]))

    def test_is_applicable(self):
        c = Context(
            items=(
                ItemAssertion(DiscreteItem('1')),
                ItemAssertion(DiscreteItem('2'), negated=True),
                ItemAssertion(ContinuousItem(np.array([1.0, 0.0]))),
                ItemAssertion(ContinuousItem(np.array([0.0, 1.0])), negated=True)
            ))

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be satisfied
        ##########################
        self.assertTrue(schema.is_applicable(state=['1', np.array([1.0, 0.0])]))

        # expected to NOT be satisfied
        ##############################

        # case 1: present negated discrete item
        self.assertFalse(schema.is_applicable(state=['1', '2', np.array([1.0, 0.0])]))

        # case 2: present negated continuous item
        self.assertFalse(schema.is_applicable(state=['1', np.array([1.0, 0.0]), np.array([0.0, 1.0])]))

        # case 3: missing discrete element
        self.assertFalse(schema.is_applicable(state=['3', np.array([1.0, 0.0])]))

        # case 4: missing continuous element
        self.assertFalse(schema.is_applicable(state=['1', np.array([0.5, 0.5])]))

        # Tests overriding conditions
        #############################
        schema.overriding_conditions = StateAssertion((ItemAssertion(DiscreteItem('3')),))

        # expected to be applicable
        self.assertTrue(schema.is_applicable(state=['1', np.array([1.0, 0.0])]))

        # expected to NOT be applicable (due to overriding condition)
        self.assertFalse(schema.is_applicable(state=['1', '3', np.array([1.0, 0.0])]))

    def test_spin_off(self):
        # test bare schema spin-off
        ###########################

        s1 = Schema(action=Action())

        # context spin-off
        s2 = s1.create_spin_off(mode='context',
                                item=ItemAssertion(DiscreteItem('1')))

        self.assertNotEqual(s1.id, s2.id)
        self.assertEqual(1, len(s2.context))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s2.context)
        self.assertIsNone(s2.result)

        # result spin-off
        s3 = s1.create_spin_off(mode='result',
                                item=ItemAssertion(DiscreteItem('1')))

        self.assertNotEqual(s1.id, s3.id)
        self.assertEqual(s1.action, s2.action)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s3.result)
        self.assertIsNone(s3.context)

        # test spin-off for schema with context, no result
        ##################################################
        s1 = Schema(action=Action(),
                    context=Context(
                        items=(
                            ItemAssertion(DiscreteItem('1')),
                            ItemAssertion(ContinuousItem(np.array([1.0, 0.0])))
                        )))

        # context spin-off
        s2 = s1.create_spin_off(mode='context',
                                item=ItemAssertion(DiscreteItem('2')))

        self.assertNotEqual(s1.id, s2.id)
        self.assertEqual(3, len(s2.context))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s2.context)
        self.assertTrue(ItemAssertion(DiscreteItem('2')) in s2.context)
        self.assertTrue(ItemAssertion(ContinuousItem(np.array([1.0, 0.0]))) in s2.context)
        self.assertIsNone(s2.result)

        # result spin-off
        s3 = s1.create_spin_off(mode='result',
                                item=ItemAssertion(DiscreteItem('1')))

        self.assertNotEqual(s1.id, s3.id)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s3.result)
        self.assertEqual(2, len(s3.context))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s3.context)
        self.assertTrue(ItemAssertion(ContinuousItem(np.array([1.0, 0.0]))) in s3.context)

        # test spin-off for schema with result, no context
        ##################################################
        s1 = Schema(action=Action(),
                    result=Result(
                        items=(
                            ItemAssertion(DiscreteItem('1')),
                            ItemAssertion(ContinuousItem(np.array([1.0, 0.0])))
                        )))

        # context spin-off
        s2 = s1.create_spin_off(mode='context',
                                item=ItemAssertion(DiscreteItem('1')))

        self.assertNotEqual(s1.id, s2.id)
        self.assertEqual(1, len(s2.context))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s2.context)
        self.assertEqual(2, len(s2.result))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s2.result)
        self.assertTrue(ItemAssertion(ContinuousItem(np.array([1.0, 0.0]))) in s2.result)

        # result spin-off
        s3 = s1.create_spin_off(mode='result',
                                item=ItemAssertion(DiscreteItem('2')))

        self.assertNotEqual(s1.id, s3.id)
        self.assertIsNone(s3.context)
        self.assertEqual(3, len(s3.result))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s3.result)
        self.assertTrue(ItemAssertion(DiscreteItem('2')) in s3.result)
        self.assertTrue(ItemAssertion(ContinuousItem(np.array([1.0, 0.0]))) in s3.result)

        # test spin-off for schema with both a context and a result
        ###########################################################
        s1 = Schema(action=Action(),
                    context=Context(
                        items=(
                            ItemAssertion(DiscreteItem('1')),
                            ItemAssertion(ContinuousItem(np.array([1.0, 0.0])))
                        )),
                    result=Result(
                        items=(
                            ItemAssertion(DiscreteItem('1')),
                            ItemAssertion(ContinuousItem(np.array([0.0, 1.0])))
                        )))

        # context spin-off
        s2 = s1.create_spin_off(mode='context',
                                item=ItemAssertion(DiscreteItem('2')))

        self.assertNotEqual(s1.id, s2.id)
        self.assertEqual(3, len(s2.context))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s2.context)
        self.assertTrue(ItemAssertion(DiscreteItem('2')) in s2.context)
        self.assertTrue(ItemAssertion(ContinuousItem(np.array([1.0, 0.0]))) in s2.context)
        self.assertEqual(2, len(s2.result))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s2.result)
        self.assertTrue(ItemAssertion(ContinuousItem(np.array([0.0, 1.0]))) in s2.result)

        # result spin-off
        s3 = s1.create_spin_off(mode='result',
                                item=ItemAssertion(DiscreteItem('2')))

        self.assertNotEqual(s1.id, s3.id)
        self.assertEqual(3, len(s3.result))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s3.result)
        self.assertTrue(ItemAssertion(DiscreteItem('2')) in s3.result)
        self.assertTrue(ItemAssertion(ContinuousItem(np.array([0.0, 1.0]))) in s3.result)
        self.assertEqual(2, len(s3.context))
        self.assertTrue(ItemAssertion(DiscreteItem('1')) in s3.context)
        self.assertTrue(ItemAssertion(ContinuousItem(np.array([1.0, 0.0]))) in s3.context)

        try:
            s3.create_spin_off(mode='result',
                               item=ItemAssertion(DiscreteItem('2')))
        except ValueError as e:
            self.assertEqual(str(e), 'ItemAssertion already exists in StateAssertion')
