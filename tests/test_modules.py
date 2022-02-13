from unittest import TestCase

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import NULL_STATE_ASSERT
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import state_primitive_value
from schema_mechanism.func_api import sym_assert
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import create_spin_off
from schema_mechanism.modules import held_state
from schema_mechanism.modules import lost_state
from schema_mechanism.modules import new_state


class TestModuleFunctions(TestCase):
    def setUp(self) -> None:
        pass

    def test_spinoff_schema(self):
        # test bare schema spin-off
        ###########################

        s1 = Schema(action=Action())

        # context spin-off
        s2 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.CONTEXT, item_assert=sym_assert('1'))

        self.assertEqual(1, len(s2.context))
        self.assertTrue(sym_assert('1') in s2.context)
        self.assertIs(NULL_STATE_ASSERT, s2.result)

        # result spin-off
        s3 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.RESULT, item_assert=sym_assert('1'))

        self.assertEqual(s1.action, s2.action)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(sym_assert('1') in s3.result)
        self.assertIs(NULL_STATE_ASSERT, s3.context)

        # test spin-off for schema with context, no result
        ##################################################
        s1 = Schema(action=Action(), context=sym_state_assert('1'))

        # context spin-off
        s2 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.CONTEXT, item_assert=sym_assert('2'))

        self.assertEqual(2, len(s2.context))
        self.assertTrue(sym_assert('1') in s2.context)
        self.assertTrue(sym_assert('2') in s2.context)
        self.assertIs(NULL_STATE_ASSERT, s2.result)

        # result spin-off
        s3 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.RESULT, item_assert=sym_assert('1'))

        self.assertEqual(1, len(s3.result))
        self.assertTrue(sym_assert('1') in s3.result)
        self.assertEqual(1, len(s3.context))
        self.assertTrue(sym_assert('1') in s3.context)

        # test spin-off for schema with result, no context
        ##################################################
        s1 = Schema(action=Action(), result=sym_state_assert('1'))

        # context spin-off
        s2 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.CONTEXT, item_assert=sym_assert('1'))

        self.assertEqual(1, len(s2.context))
        self.assertTrue(sym_assert('1') in s2.context)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(sym_assert('1') in s2.result)

        # result spin-off
        s3 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.RESULT, item_assert=sym_assert('2'))

        self.assertIs(NULL_STATE_ASSERT, s3.context)
        self.assertEqual(2, len(s3.result))
        self.assertTrue(sym_assert('1') in s3.result)
        self.assertTrue(sym_assert('2') in s3.result)

        # test spin-off for schema with both a context and a result
        ###########################################################
        s1 = Schema(action=Action(),
                    context=sym_state_assert('1'),
                    result=sym_state_assert('1'))

        # context spin-off
        s2 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.CONTEXT, item_assert=sym_assert('2'))

        self.assertEqual(2, len(s2.context))
        self.assertTrue(sym_assert('1') in s2.context)
        self.assertTrue(sym_assert('2') in s2.context)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(sym_assert('1') in s2.result)

        # result spin-off
        s3 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.RESULT, item_assert=sym_assert('2'))

        self.assertEqual(2, len(s3.result))
        self.assertTrue(sym_assert('1') in s3.result)
        self.assertTrue(sym_assert('2') in s3.result)
        self.assertEqual(1, len(s3.context))
        self.assertTrue(sym_assert('1') in s3.context)

        try:
            create_spin_off(schema=s3, spin_off_type=Schema.SpinOffType.CONTEXT, item_assert=sym_assert('1'))
        except ValueError as e:
            self.assertEqual(str(e), 'ItemAssertion already exists in StateAssertion')

        try:
            create_spin_off(schema=s3, spin_off_type=Schema.SpinOffType.RESULT, item_assert=sym_assert('2'))
        except ValueError as e:
            self.assertEqual(str(e), 'ItemAssertion already exists in StateAssertion')


class TestStateFunctions(TestCase):
    def setUp(self) -> None:
        self.s_1 = sym_state('1,2,3,4,5')
        self.s_2 = sym_state('4,5,6,7,8')
        self.s_empty = []
        self.s_none = None

    def test_held_state(self):
        self.assertEqual(0, len(held_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(held_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(held_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(held_state(s_prev=self.s_none, s_curr=self.s_none)))

        self.assertSetEqual(set(sym_state('4,5')), held_state(s_prev=self.s_1, s_curr=self.s_2))
        self.assertSetEqual(set(sym_state('4,5')), held_state(s_prev=self.s_2, s_curr=self.s_1))

    def test_lost_state(self):
        self.assertEqual(0, len(lost_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_none, s_curr=self.s_none)))

        self.assertSetEqual(set(sym_state('1,2,3')), lost_state(s_prev=self.s_1, s_curr=self.s_2))
        self.assertSetEqual(set(sym_state('6,7,8')), lost_state(s_prev=self.s_2, s_curr=self.s_1))

    def test_new_state(self):
        self.assertEqual(0, len(new_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(new_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(new_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(new_state(s_prev=self.s_none, s_curr=self.s_none)))

        self.assertSetEqual(set(sym_state('6,7,8')), new_state(s_prev=self.s_1, s_curr=self.s_2))
        self.assertSetEqual(set(sym_state('1,2,3')), new_state(s_prev=self.s_2, s_curr=self.s_1))

    def test_state_primitive_value(self):
        # adds items to item pool
        i1 = sym_item('1', -1)
        i2 = sym_item('2', 0)
        i3 = sym_item('3', 1)

        # empty state
        self.assertEqual(0.0, state_primitive_value(sym_state('')))

        # single element state (negative primitive value)
        self.assertEqual(-1.0, state_primitive_value(sym_state('1')))

        # single element state (zero primitive value)
        self.assertEqual(0.0, state_primitive_value(sym_state('2')))

        # single element state (positive primitive value)
        self.assertEqual(1.0, state_primitive_value(sym_state('3')))

        # multiple element state (negative, zero, and positive primitive values)
        self.assertEqual(-1.0, state_primitive_value(sym_state('1,2')))
        self.assertEqual(1.0, state_primitive_value(sym_state('2,3')))
        self.assertEqual(0.0, state_primitive_value(sym_state('1,3')))
        self.assertEqual(0.0, state_primitive_value(sym_state('1,2,3')))
