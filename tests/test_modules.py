from unittest import TestCase

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Context
from schema_mechanism.data_structures import Result
from schema_mechanism.data_structures import Schema
from schema_mechanism.func_api import make_assertion
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
        s2 = create_spin_off(schema=s1, mode='context', item_assert=make_assertion('1'))

        self.assertEqual(1, len(s2.context))
        self.assertTrue(make_assertion('1') in s2.context)
        self.assertIsNone(s2.result)

        # result spin-off
        s3 = create_spin_off(schema=s1, mode='result', item_assert=make_assertion('1'))

        self.assertEqual(s1.action, s2.action)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(make_assertion('1') in s3.result)
        self.assertIsNone(s3.context)

        # test spin-off for schema with context, no result
        ##################################################
        s1 = Schema(action=Action(),
                    context=Context(item_asserts=(make_assertion('1'),)))

        # context spin-off
        s2 = create_spin_off(schema=s1, mode='context', item_assert=make_assertion('2'))

        self.assertEqual(2, len(s2.context))
        self.assertTrue(make_assertion('1') in s2.context)
        self.assertTrue(make_assertion('2') in s2.context)
        self.assertIsNone(s2.result)

        # result spin-off
        s3 = create_spin_off(schema=s1, mode='result', item_assert=make_assertion('1'))

        self.assertEqual(1, len(s3.result))
        self.assertTrue(make_assertion('1') in s3.result)
        self.assertEqual(1, len(s3.context))
        self.assertTrue(make_assertion('1') in s3.context)

        # test spin-off for schema with result, no context
        ##################################################
        s1 = Schema(action=Action(),
                    result=Result(
                        item_asserts=(make_assertion('1'),)))

        # context spin-off
        s2 = create_spin_off(schema=s1, mode='context', item_assert=make_assertion('1'))

        self.assertEqual(1, len(s2.context))
        self.assertTrue(make_assertion('1') in s2.context)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(make_assertion('1') in s2.result)

        # result spin-off
        s3 = create_spin_off(schema=s1, mode='result', item_assert=make_assertion('2'))

        self.assertIsNone(s3.context)
        self.assertEqual(2, len(s3.result))
        self.assertTrue(make_assertion('1') in s3.result)
        self.assertTrue(make_assertion('2') in s3.result)

        # test spin-off for schema with both a context and a result
        ###########################################################
        s1 = Schema(action=Action(),
                    context=Context(item_asserts=(make_assertion('1'),)),
                    result=Result(item_asserts=(make_assertion('1'),)))

        # context spin-off
        s2 = create_spin_off(schema=s1, mode='context', item_assert=make_assertion('2'))

        self.assertEqual(2, len(s2.context))
        self.assertTrue(make_assertion('1') in s2.context)
        self.assertTrue(make_assertion('2') in s2.context)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(make_assertion('1') in s2.result)

        # result spin-off
        s3 = create_spin_off(schema=s1, mode='result', item_assert=make_assertion('2'))

        self.assertEqual(2, len(s3.result))
        self.assertTrue(make_assertion('1') in s3.result)
        self.assertTrue(make_assertion('2') in s3.result)
        self.assertEqual(1, len(s3.context))
        self.assertTrue(make_assertion('1') in s3.context)

        try:
            create_spin_off(schema=s3, mode='result', item_assert=make_assertion('2'))
        except ValueError as e:
            self.assertEqual(str(e), 'ItemAssertion already exists in StateAssertion')


class TestStateFunctions(TestCase):
    def setUp(self) -> None:
        self.s_1 = [1, 2, 3, 4, 5]
        self.s_2 = [4, 5, 6, 7, 8]
        self.s_empty = []
        self.s_none = None

    def test_held_state(self):
        self.assertEqual(0, len(held_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(held_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(held_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(held_state(s_prev=self.s_none, s_curr=self.s_none)))

        self.assertSetEqual({4, 5}, held_state(s_prev=self.s_1, s_curr=self.s_2))
        self.assertSetEqual({4, 5}, held_state(s_prev=self.s_2, s_curr=self.s_1))

    def test_lost_state(self):
        self.assertEqual(0, len(lost_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_none, s_curr=self.s_none)))

        self.assertSetEqual({1, 2, 3}, lost_state(s_prev=self.s_1, s_curr=self.s_2))
        self.assertSetEqual({6, 7, 8}, lost_state(s_prev=self.s_2, s_curr=self.s_1))

    def test_new_state(self):
        self.assertEqual(0, len(new_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(new_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(new_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(new_state(s_prev=self.s_none, s_curr=self.s_none)))

        self.assertSetEqual({6, 7, 8}, new_state(s_prev=self.s_1, s_curr=self.s_2))
        self.assertSetEqual({1, 2, 3}, new_state(s_prev=self.s_2, s_curr=self.s_1))