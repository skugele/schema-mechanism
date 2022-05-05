from unittest import TestCase

from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.modules import create_spin_off
from test_share.test_func import common_test_setup


class TestModuleFunctions(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_spinoff_schema(self):
        # test bare schema spin-off
        ###########################

        s1 = sym_schema('/action/')

        # result spin-off from item assertion
        s2 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.RESULT, item=sym_item('1'))

        self.assertEqual(s1.action, s2.action)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(sym_item('1') in s2.result)
        self.assertIs(NULL_STATE_ASSERT, s2.context)

        # result spin-off from item assertion with composite item
        s3 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.RESULT, item=sym_item('(1,2)'))

        self.assertEqual(s1.action, s3.action)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(sym_item('(1,2)') in s3.result)
        self.assertIs(NULL_STATE_ASSERT, s3.context)

        s_non_prim_1 = sym_schema('/action/11,12')
        s_non_prim_2 = sym_schema('10,/action/(1,2),')

        # test: result spin-offs MUST originate from primitive schemas
        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_1,
                                                              spin_off_type=Schema.SpinOffType.RESULT,
                                                              item=sym_item('3')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_2,
                                                              spin_off_type=Schema.SpinOffType.RESULT,
                                                              item=sym_item('3')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_1,
                                                              spin_off_type=Schema.SpinOffType.RESULT,
                                                              item=sym_item('(3,4)')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_2,
                                                              spin_off_type=Schema.SpinOffType.RESULT,
                                                              item=sym_item('(3,4)')))

    def test_spin_off_for_composite_action_schemas(self):
        # test bare schema spin-off
        ###########################

        s1 = sym_schema('/A,B,C,1,2,3/')

        # result spin-off from item assertion
        s2 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.RESULT, item=sym_item('1'))

        self.assertEqual(s1.action, s2.action)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(sym_item('1') in s2.result)
        self.assertIs(NULL_STATE_ASSERT, s2.context)

        # result spin-off from item assertion with composite item
        s3 = create_spin_off(schema=s1, spin_off_type=Schema.SpinOffType.RESULT, item=sym_item('(1,2)'))

        self.assertEqual(s1.action, s3.action)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(sym_item('(1,2)') in s3.result)
        self.assertIs(NULL_STATE_ASSERT, s3.context)

        s_non_prim_1 = sym_schema('/action/11,12')
        s_non_prim_2 = sym_schema('10,/action/(1,2),')

        # test: result spin-offs MUST originate from primitive schemas (unless ER_INCREMENTAL_RESULTS enabled)
        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_1,
                                                              spin_off_type=Schema.SpinOffType.RESULT,
                                                              item=sym_item('3')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_2,
                                                              spin_off_type=Schema.SpinOffType.RESULT,
                                                              item=sym_item('3')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_1,
                                                              spin_off_type=Schema.SpinOffType.RESULT,
                                                              item=sym_item('(3,4)')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_2,
                                                              spin_off_type=Schema.SpinOffType.RESULT,
                                                              item=sym_item('(3,4)')))
