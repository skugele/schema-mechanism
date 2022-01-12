from unittest import TestCase

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Context
from schema_mechanism.data_structures import Result
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.func_api import make_assertion


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

        # Verify immutability
        s = Schema(context=Context(item_asserts=(make_assertion('1'),)),
                   action=Action(),
                   result=Result(item_asserts=(make_assertion('2'),)))

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
            make_assertion('1'),
            make_assertion('2', negated=True),
            make_assertion('3')
        ))

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be satisfied
        ##########################
        self.assertTrue(schema.context.is_satisfied(state=['1', '3']))
        self.assertTrue(schema.context.is_satisfied(state=['1', '3', '4']))

        # expected to NOT be satisfied
        ##############################
        # case 1: present negated item
        self.assertFalse(schema.context.is_satisfied(state=['1', '2', '3']))

        # case 2: missing non-negated item
        self.assertFalse(schema.context.is_satisfied(state=['1']))
        self.assertFalse(schema.context.is_satisfied(state=['3']))

        # case 3 : both present negated item and missing non-negated item
        self.assertFalse(schema.context.is_satisfied(state=['1', '2']))
        self.assertFalse(schema.context.is_satisfied(state=['2', '3']))

    def test_is_applicable(self):
        c = Context(
            item_asserts=(
                make_assertion('1'),
                make_assertion('2', negated=True),
                make_assertion('3'),
            ))

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be applicable
        ##########################
        self.assertTrue(schema.is_applicable(state=['1', '3']))
        self.assertTrue(schema.is_applicable(state=['1', '3', '4']))

        # expected to NOT be applicable
        ###############################

        # case 1: present negated item
        self.assertFalse(schema.is_applicable(state=['1', '2', '3']))

        # case 2: missing non-negated item
        self.assertFalse(schema.is_applicable(state=['1']))
        self.assertFalse(schema.is_applicable(state=['3']))

        # case 3 : both present negated item and missing non-negated item
        self.assertFalse(schema.is_applicable(state=['1', '2']))
        self.assertFalse(schema.is_applicable(state=['2', '3']))

        # Tests overriding conditions
        #############################
        schema.overriding_conditions = StateAssertion((make_assertion('5'),))

        # expected to be applicable
        self.assertTrue(schema.is_applicable(state=['1', '3', '4']))

        # expected to NOT be applicable (due to overriding condition)
        self.assertFalse(schema.is_applicable(state=['1', '3', '4', '5']))

    def test_spin_off(self):
        # test bare schema spin-off
        ###########################

        s1 = Schema(action=Action())

        # context spin-off
        s2 = s1.create_spin_off(mode='context', item_assert=make_assertion('1'))

        self.assertEqual(1, len(s2.context))
        self.assertTrue(make_assertion('1') in s2.context)
        self.assertIsNone(s2.result)

        # result spin-off
        s3 = s1.create_spin_off(mode='result', item_assert=make_assertion('1'))

        self.assertEqual(s1.action, s2.action)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(make_assertion('1') in s3.result)
        self.assertIsNone(s3.context)

        # test spin-off for schema with context, no result
        ##################################################
        s1 = Schema(action=Action(),
                    context=Context(item_asserts=(make_assertion('1'),)))

        # context spin-off
        s2 = s1.create_spin_off(mode='context', item_assert=make_assertion('2'))

        self.assertEqual(2, len(s2.context))
        self.assertTrue(make_assertion('1') in s2.context)
        self.assertTrue(make_assertion('2') in s2.context)
        self.assertIsNone(s2.result)

        # result spin-off
        s3 = s1.create_spin_off(mode='result', item_assert=make_assertion('1'))

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
        s2 = s1.create_spin_off(mode='context', item_assert=make_assertion('1'))

        self.assertEqual(1, len(s2.context))
        self.assertTrue(make_assertion('1') in s2.context)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(make_assertion('1') in s2.result)

        # result spin-off
        s3 = s1.create_spin_off(mode='result',
                                item_assert=make_assertion('2'))

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
        s2 = s1.create_spin_off(mode='context', item_assert=make_assertion('2'))

        self.assertEqual(2, len(s2.context))
        self.assertTrue(make_assertion('1') in s2.context)
        self.assertTrue(make_assertion('2') in s2.context)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(make_assertion('1') in s2.result)

        # result spin-off
        s3 = s1.create_spin_off(mode='result', item_assert=make_assertion('2'))

        self.assertEqual(2, len(s3.result))
        self.assertTrue(make_assertion('1') in s3.result)
        self.assertTrue(make_assertion('2') in s3.result)
        self.assertEqual(1, len(s3.context))
        self.assertTrue(make_assertion('1') in s3.context)

        try:
            s3.create_spin_off(mode='result', item_assert=make_assertion('2'))
        except ValueError as e:
            self.assertEqual(str(e), 'ItemAssertion already exists in StateAssertion')
