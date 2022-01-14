from unittest import TestCase

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Context
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import Result
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import create_item
from schema_mechanism.func_api import make_assertion
from schema_mechanism.func_api import make_assertions
from test_share.test_classes import MockObserver


class TestSchema(TestCase):
    def setUp(self) -> None:
        self._item_pool = ItemPool()

        # populate pool
        for i in range(10):
            _ = self._item_pool.get(i, SymbolicItem)

        self.obs = MockObserver()

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

    def test_update(self):
        s = Schema(context=Context(make_assertions([1])),
                   action=Action(),
                   result=Result(make_assertions([2])))
        s.register(self.obs)

        s_a = [0, 1, 2]

        # TODO: Uncomment this once the "held" items list is operational
        # TODO: Update these test cases once context/result suppression for item relevance determination is in place
        # s_na = [0, 1, 2]

        s_na = [4, 5, 6]

        s.update(activated=True, view=ItemPoolStateView(s_a), count=10)

        # for k, v in s.extended_result.stats.items():
        #     print(f'{k} -> {repr(v)}')

        for se in s_a:
            i_stats = s.extended_context.stats.get(create_item(se))

            self.assertEqual(10, i_stats.n_on)

        s.update(activated=False, view=ItemPoolStateView(s_na), count=1)

        for se in s_na:
            i_stats = s.extended_context.stats.get(create_item(se))

            self.assertEqual(i_stats.n_off, 10)

        # verify observer notified
        self.assertTrue(self.obs.n_received >= 1)

    def test_notify_all(self):
        s = Schema(context=Context(make_assertions([1, 2, 3])),
                   action=Action(),
                   result=Result(make_assertions([1, 2, 3, 4])))

        state = [1, 2, 3, 4, 5, 6]
        view = ItemPoolStateView(state)

        s.update(activated=True,
                 view=view)

    # def test_performance(self):
    #     s = Schema(context=Context(make_assertions([1])),
    #                action=Action(),
    #                result=Result(make_assertions([2])))
    #     s.register(self.obs)
    #
    #     n_states = 32
    #     for _ in range(n_states):
    #         state = sample(range(len(self._item_pool)), k=3)
    #
    #         if random.randint(0, 3) == 0:
    #             s.update(activated=True, view=ItemPoolStateView(state))
    #         else:
    #             s.update(activated=False, view=ItemPoolStateView(state))
    #
    #     for v in s.extended_result.stats.values():
    #         print(repr(v))
