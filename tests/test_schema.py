import os
from copy import copy
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from random import sample
from tempfile import TemporaryDirectory
from time import time
from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import test_share
from schema_mechanism.core import Action
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import ItemPool
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaSpinOffType
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem
from schema_mechanism.core import get_global_params
from schema_mechanism.core import is_reliable
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.func_api import update_schema
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import serialize
from schema_mechanism.share import SupportedFeature
from schema_mechanism.strategies.correlation_test import DrescherCorrelationTest
from schema_mechanism.util import repr_str
from test_share.test_classes import MockObserver
from test_share.test_classes import MockSchema
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestSchema(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.item_pool = ItemPool()

        # populate pool
        for i in range(10):
            _ = self.item_pool.get(str(i), primitive_value=1.0, item_type=SymbolicItem)

        self.schema = sym_schema('0,1/A1/2,3,4')
        self.schema_ca = sym_schema('0,1/2,3/2,3,4')

        self.obs = MockObserver()
        self.schema.register(self.obs)
        self.schema_ca.register(self.obs)

        params = get_global_params()
        features: set[SupportedFeature] = params.get('features')

        if SupportedFeature.EC_SUPPRESS_UPDATE_ON_RELIABLE in features:
            features.remove(SupportedFeature.EC_SUPPRESS_UPDATE_ON_RELIABLE)

        if SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED in features:
            features.remove(SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED)

    def test_init(self):
        # Action CANNOT be None
        try:
            # noinspection PyTypeChecker
            Schema(action=None)
            self.fail('Action=None should generate a ValueError')
        except ValueError as e:
            self.assertEqual('Action cannot be None', str(e))

        # Context and Result CAN be None
        try:
            s = Schema(action=Action('My Action'))
            self.assertIs(s.context, NULL_STATE_ASSERT)
            self.assertIs(s.result, NULL_STATE_ASSERT)
        except Exception as e:
            self.fail(f'Unexpected exception raised: {e}')

        # Verify immutability
        s = Schema(context=sym_state_assert('1,'),
                   action=Action(),
                   result=sym_state_assert('2,'))

        try:
            # noinspection PyPropertyAccess
            s.context = StateAssertion()
            self.fail('Schema\'s context is not immutable as expected')
        except AttributeError:
            pass

        try:
            # noinspection PyPropertyAccess
            s.result = StateAssertion()
            self.fail('Schema\'s result is not immutable as expected')
        except AttributeError:
            pass

        try:
            # noinspection PyPropertyAccess
            s.action = Action()
            self.fail('Schema\'s action is not immutable as expected')
        except AttributeError:
            pass

        # test: creation time should be set and bounded by the times before and after schema creation
        before = time()
        s = Schema(action=Action())
        after = time()

        self.assertIsNotNone(s.creation_time)
        self.assertGreaterEqual(s.creation_time, before)
        self.assertLessEqual(s.creation_time, after)

    def test_cost(self):
        # test: cost getter/setter should correctly set the value
        cost = 17.2
        self.schema.cost = cost

        self.assertEqual(cost, self.schema.cost)

    def test_repr(self):
        # test: repr should return the expected str representation
        expected_repr = repr_str(self.schema, {'uid': self.schema.uid,
                                               'creation_time': datetime.fromtimestamp(self.schema.creation_time),
                                               'context': self.schema.context,
                                               'cost': self.schema.cost,
                                               'action': self.schema.action,
                                               'result': self.schema.result,
                                               'overriding_conditions': self.schema.overriding_conditions,
                                               'reliability': self.schema.reliability, })

        self.assertEqual(expected_repr, repr(self.schema))

    # Schema.update
    # 	conditions to test:
    #             if is_feature_enabled(SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED) and explained:
    #                 logger.debug(
    #                     f'update suppressed for schema {self} because its result was explained by a reliable schema')
    #             else:
    #                 self._extended_result.update_all(activated=activated, new=new, lost=lost, count=count)
    #
    # Schema.receive (test unrecognized spin_off_type raises a ValueError)

    def test_is_context_satisfied(self):
        c = sym_state_assert('1,3')
        schema = Schema(context=c, action=Action(), result=None)

        # expected to be satisfied
        ##########################
        self.assertTrue(schema.context.is_satisfied(state=sym_state('1,3')))
        self.assertTrue(schema.context.is_satisfied(state=sym_state('1,3,4')))

        # expected to NOT be satisfied
        ##############################
        # case 1: missing item
        self.assertFalse(schema.context.is_satisfied(state=sym_state('1')))
        self.assertFalse(schema.context.is_satisfied(state=sym_state('3')))

        # complex action schema
        self.assertTrue(self.schema_ca.context.is_satisfied(state=sym_state('0,1')))
        self.assertTrue(self.schema_ca.context.is_satisfied(state=sym_state('0,1,2')))

    def test_is_applicable(self):
        # test: bare schemas should always be applicable
        schema = Schema(action=Action())
        self.assertTrue(schema.is_applicable(state=sym_state('1')))

        c = sym_state_assert('1,3')

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be applicable
        ##########################
        self.assertTrue(schema.is_applicable(state=sym_state('1,3')))
        self.assertTrue(schema.is_applicable(state=sym_state('1,3,4')))

        # expected to NOT be applicable
        ###############################

        # case 1: missing item
        self.assertFalse(schema.is_applicable(state=sym_state('1')))
        self.assertFalse(schema.is_applicable(state=sym_state('3')))

        # Tests overriding conditions
        #############################
        schema.overriding_conditions = sym_state_assert('5,')

        # expected to be applicable
        self.assertTrue(schema.is_applicable(state=sym_state('1,3,4')))

        # expected to NOT be applicable (due to overriding condition)
        self.assertFalse(schema.is_applicable(state=sym_state('1,3,4,5')))

    def test_is_applicable_with_composite_action(self):
        # Tests applicability of schema with complex action
        ###################################################

        # test: if context is not satisfied then it SHOULD NOT be applicable
        self.assertFalse(self.schema_ca.is_applicable(sym_state('A,B')))
        self.assertFalse(self.schema_ca.is_applicable(sym_state('0')))
        self.assertFalse(self.schema_ca.is_applicable(sym_state('1')))

        # test: schema SHOULD NOT be applicable if its action is not enabled
        with patch('schema_mechanism.core.CompositeAction.is_enabled', return_value=False) as mock:
            self.assertFalse(self.schema_ca.is_applicable(sym_state('0,1')))
            mock.assert_called()

        # test: schema SHOULD be applicable if its action is enabled and its context is satisfied
        with patch('schema_mechanism.core.CompositeAction.is_enabled', return_value=True) as mock:
            self.assertTrue(self.schema_ca.is_applicable(sym_state('0,1')))
            mock.assert_called()

    def test_is_activated(self):
        schema_blank_context = sym_schema('/A1/C,')
        schema_a = sym_schema('A,/A1/C,')
        schema_ab = sym_schema('A,B/A2/C,D')
        schema_cd = sym_schema('C,D/A1/S1,')

        composite_action_schema = sym_schema('/S1,/C,')

        schemas = [
            schema_blank_context,
            schema_a,
            schema_ab,
            schema_cd,
            composite_action_schema
        ]

        state_a = sym_state('A')
        state_ab = sym_state('A,B')
        state_cde = sym_state('C,D,E')
        state_s1 = sym_state('S1')

        states = [
            state_a,
            state_ab,
            state_cde,
            state_s1
        ]

        # test: should return False for non-applicable schemas
        explicitly_activated_schema = schema_a
        for state in states:
            for schema in schemas:
                is_applicable = schema.is_applicable(state)
                if not is_applicable:
                    is_activated = schema.is_activated(
                        schema=explicitly_activated_schema,
                        state=state,
                        applicable=is_applicable
                    )
                    self.assertFalse(is_activated)

        # test: should return False if action not the same as explicitly activated schema
        explicitly_activated_schema = schema_a
        for state in states:
            for schema in schemas:
                is_applicable = schema.is_applicable(state)
                if schema.action != explicitly_activated_schema.action:
                    is_activated = schema.is_activated(
                        schema=explicitly_activated_schema,
                        state=state,
                        applicable=is_applicable
                    )
                    self.assertFalse(is_activated)

        # test: non-composite action schemas should return True if applicable and share same action as explicitly
        #     : activated schema
        explicitly_activated_schema = schema_a
        for state in states:
            for schema in schemas:
                if schema.action == explicitly_activated_schema.action and not schema.action.is_composite():
                    is_activated = schema.is_activated(
                        schema=explicitly_activated_schema,
                        state=state,
                        applicable=True
                    )
                    self.assertTrue(is_activated)

        # test: should return True if the schema is the explicitly activated schema
        for explicitly_activated_schema in schemas:
            for schema in schemas:
                if schema is explicitly_activated_schema:
                    is_activated = schema.is_activated(
                        schema=explicitly_activated_schema,
                        state=sym_state('A,B,C,D,S1'),
                        applicable=True
                    )
                    self.assertTrue(is_activated)

        # test: composite action schemas should return True whenever they are applicable and their goal state is
        #     : satisfied and False otherwise
        goal_state = composite_action_schema.action.goal_state
        for explicitly_activated_schema in schemas:
            for state in states:
                is_activated = composite_action_schema.is_activated(
                    schema=explicitly_activated_schema,
                    state=state,
                    applicable=True
                )
                if goal_state.is_satisfied(state):
                    self.assertTrue(is_activated)

    def test_predicts_state(self):
        state = sym_state('1,2,3')  # used for all test cases

        # test: should return False when schema has blank result
        self.assertFalse(sym_schema('/action/').predicts_state(state))

        # test: should only return True when schema's result's positive assertions cover all state elements

        # ---> positive cases
        self.assertTrue(sym_schema('/action/1,2,3').predicts_state(state))

        # ---> negative cases
        self.assertFalse(sym_schema('/action/1,2,4').predicts_state(state))
        self.assertFalse(sym_schema('/action/2,3').predicts_state(state))
        self.assertFalse(sym_schema('/action/2,').predicts_state(state))

        # test: non-composite items within composite items must also be accounted for

        # ---> positive cases
        self.assertTrue(sym_schema('/action/(1,2,3),').predicts_state(state))

        # ---> negative cases
        self.assertFalse(sym_schema('/action/(1,3),').predicts_state(state))

        # test: mixed results (w/ composite and non-composite items) should also be supported
        #     (note: this functionality is not currently used by the schema mechanism)

        # ---> positive cases
        self.assertTrue(sym_schema('/action/(1,2),3').predicts_state(state))

        # ---> negative cases
        self.assertFalse(sym_schema('/action/(1,2,3),4').predicts_state(state))

        # complex action schema
        self.assertTrue(self.schema_ca.predicts_state(sym_state('2,3,4')))

    def test_update_when_explained(self):
        params = get_global_params()

        features: set[SupportedFeature] = params.get('features')

        # test case depends on this feature being enabled
        features.add(SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED)

        # only bare schemas have an extended result
        schema = sym_schema('/A1/')

        # test: extended result item stats not updated if result explained (by a reliable schema)
        ext_result_before = deepcopy(schema.extended_result)
        schema.update(
            activated=True,
            succeeded=True,
            selection_state=sym_state('1,2,3'),
            new=[self.item_pool.get('1')],
            lost=[self.item_pool.get('4')],
            explained=True,
        )
        ext_result_after = deepcopy(schema.extended_result)

        # check item stats
        self.assertEqual(ext_result_before, ext_result_after)

        # test: extended result item stats SHOULD be updated if result not explained (by a reliable schema)
        ext_result_before = deepcopy(schema.extended_result)
        schema.update(
            activated=True,
            succeeded=True,
            selection_state=sym_state('1,2,3'),
            new=[self.item_pool.get('1')],
            lost=[self.item_pool.get('4')],
            explained=False,
        )
        ext_result_after = deepcopy(schema.extended_result)

        # check item stats
        self.assertNotEqual(ext_result_before, ext_result_after)

        # test case depends on this feature being enabled
        features.remove(SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED)

        # test: update should always occur (regardless of explained) when ER_SUPPRESS_UPDATE_ON_EXPLAINED is disabled
        # test: extended result item stats SHOULD be updated if result not explained (by a reliable schema)
        ext_result_before = deepcopy(schema.extended_result)
        schema.update(
            activated=True,
            succeeded=True,
            selection_state=sym_state('1,2,3'),
            new=[self.item_pool.get('1')],
            lost=[self.item_pool.get('4')],
            explained=True,
        )
        ext_result_after = deepcopy(schema.extended_result)

        # check item stats
        self.assertNotEqual(ext_result_before, ext_result_after)

    def test_update_when_ec_suppress_update_on_reliable_disabled(self):
        params = get_global_params()
        features: set[SupportedFeature] = params.get('features')

        # test case depends on this feature being disabled
        if SupportedFeature.EC_SUPPRESS_UPDATE_ON_RELIABLE in features:
            features.remove(SupportedFeature.EC_SUPPRESS_UPDATE_ON_RELIABLE)

        reliable_schema = sym_schema('1,/A1/2,', schema_type=MockSchema, reliability=1.0)
        unreliable_schema = sym_schema('3,/A2/4,', schema_type=MockSchema, reliability=0.0)

        # sanity checks: reliable_schema should be reliable; unreliable_schema should be unreliable
        self.assertTrue(is_reliable(reliable_schema))
        self.assertFalse(is_reliable(unreliable_schema))

        # test: update should always occur (regardless of explained) when EC_SUPPRESS_UPDATE_ON_RELIABLE is disabled
        for schema in [reliable_schema, unreliable_schema]:
            ext_context_before = deepcopy(schema.extended_context)
            schema.update(
                activated=True,
                succeeded=True,
                selection_state=sym_state('1,2,3'),
                new=[self.item_pool.get('1')],
                lost=[self.item_pool.get('4')],
                explained=False,
            )
            ext_context_after = deepcopy(schema.extended_context)

            # check item stats
            self.assertNotEqual(ext_context_before, ext_context_after)

    def test_update_when_ec_suppress_update_on_reliable_enabled(self):
        params = get_global_params()
        features: set[SupportedFeature] = params.get('features')

        # test case depends on this feature being enabled
        features.add(SupportedFeature.EC_SUPPRESS_UPDATE_ON_RELIABLE)

        schema = sym_schema('X,/A1/Y,', schema_type=MockSchema, reliability=1.0)

        # sanity check: schema should be reliable
        self.assertTrue(is_reliable(schema))

        # test: extended context item stats not updated if result explained (by a reliable schema)
        ext_context_before = deepcopy(schema.extended_context)
        schema.update(
            activated=True,
            succeeded=True,
            selection_state=sym_state('1,2,3'),
            new=[self.item_pool.get('1')],
            lost=[self.item_pool.get('4')],
            explained=False,
        )
        ext_context_after = deepcopy(schema.extended_context)

        # check item stats
        self.assertEqual(ext_context_before, ext_context_after)

        schema = sym_schema('A,/A2/B,', schema_type=MockSchema, reliability=0.0)

        # sanity check: schema should be unreliable
        self.assertFalse(is_reliable(schema))

        # test: extended context item stats SHOULD be updated if schema not reliable
        ext_context_before = deepcopy(schema.extended_context)
        schema.update(
            activated=True,
            succeeded=True,
            selection_state=sym_state('1,2,3'),
            new=[self.item_pool.get('1')],
            lost=[self.item_pool.get('4')],
            explained=False,
        )
        ext_context_after = deepcopy(schema.extended_context)

        # check item stats
        self.assertNotEqual(ext_context_before, ext_context_after)

    def test_update_when_activated_and_successful(self):
        s_prev = sym_state('0,1,2,3')
        s_curr = sym_state('2,3,4,5')

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=True, succeeded=True, selection_state=s_prev, new=new, lost=lost)

        # test: total number of updates (n) SHOULD always be updated
        self.assertEqual(1, self.schema.stats.n)

        # test: n_success SHOULD be updated on success
        self.assertEqual(1, self.schema.stats.n_success)

        # test: n_fail SHOULD NOT be updated on success
        self.assertEqual(0, self.schema.stats.n_fail)

        # test: n_activated SHOULD be updated when schema activated
        self.assertEqual(1, self.schema.stats.n_activated)

        # check item level statistics
        for i in self.item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            # test: extended context success statistics SHOULD be updated when item is On or Off in selection state
            if i.is_on(s_prev):
                self.assertEqual(1, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(1, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)

            if self.schema.extended_result:
                er_stats = self.schema.extended_result.stats.get(i)

                # test: extended-result, activation statistics SHOULD be updated when item is in new or lost
                if i in new:
                    self.assertEqual(1, er_stats.n_on_and_activated)
                    self.assertEqual(0, er_stats.n_on_and_not_activated)
                    self.assertEqual(0, er_stats.n_off_and_activated)
                    self.assertEqual(0, er_stats.n_off_and_not_activated)
                elif i in lost:
                    self.assertEqual(0, er_stats.n_on_and_activated)
                    self.assertEqual(0, er_stats.n_on_and_not_activated)
                    self.assertEqual(1, er_stats.n_off_and_activated)
                    self.assertEqual(0, er_stats.n_off_and_not_activated)

    def test_update_when_activated_and_failure(self):
        selection_state = sym_state('0,1,2,3')
        result_state = sym_state('2,5')

        new = new_state(selection_state, result_state)
        lost = lost_state(selection_state, result_state)

        self.schema.update(activated=True, succeeded=False, selection_state=selection_state, new=new, lost=lost)

        # test: total number of updates (n) SHOULD always be updated
        self.assertEqual(1, self.schema.stats.n)

        # test: n_success SHOULD NOT be updated on failure
        self.assertEqual(0, self.schema.stats.n_success)

        # test: n_fail SHOULD be updated on failure
        self.assertEqual(1, self.schema.stats.n_fail)

        # test: n_activated SHOULD be updated when schema activated
        self.assertEqual(1, self.schema.stats.n_activated)

        # check item level statistics
        for i in self.item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            # test: extended context failure statistics SHOULD be updated when item is On or Off in selection state
            if i.is_on(selection_state):
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(1, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(1, ec_stats.n_fail_and_off)

            # test: extended-result, activation statistics SHOULD be updated when item is in new or lost
            if self.schema.extended_result:
                er_stats = self.schema.extended_result.stats.get(i)

                if i in new:
                    self.assertEqual(1, er_stats.n_on_and_activated)
                    self.assertEqual(0, er_stats.n_on_and_not_activated)
                    self.assertEqual(0, er_stats.n_off_and_activated)
                    self.assertEqual(0, er_stats.n_off_and_not_activated)
                elif i in lost:
                    self.assertEqual(0, er_stats.n_on_and_activated)
                    self.assertEqual(0, er_stats.n_on_and_not_activated)
                    self.assertEqual(1, er_stats.n_off_and_activated)
                    self.assertEqual(0, er_stats.n_off_and_not_activated)

    def test_update_when_schema_not_activated(self):
        selection_state = sym_state('0,3')
        result_state = sym_state('2,4,5')

        new = new_state(selection_state, result_state)
        lost = lost_state(selection_state, result_state)

        self.schema.update(activated=False, succeeded=False, selection_state=selection_state, new=new, lost=lost)

        # test: total number of updates (n) SHOULD always be updated
        self.assertEqual(1, self.schema.stats.n)

        # test: success/failure statistics SHOULD NOT be updated if not activated
        self.assertEqual(0, self.schema.stats.n_success)
        self.assertEqual(0, self.schema.stats.n_fail)

        # test: n_activated SHOULD NOT be updated when schema not activated
        self.assertEqual(0, self.schema.stats.n_activated)

        # check item level statistics
        for i in self.item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            # test: extended context item statistics SHOULD NOT have been updated if schema not activated
            self.assertEqual(0, ec_stats.n_success_and_on)
            self.assertEqual(0, ec_stats.n_success_and_off)
            self.assertEqual(0, ec_stats.n_fail_and_on)
            self.assertEqual(0, ec_stats.n_fail_and_off)

            if self.schema.extended_result:
                er_stats = self.schema.extended_result.stats.get(i)

                # test: extended result item statistics SHOULD be updated if item was in new or lost
                if i in new:
                    self.assertEqual(0, er_stats.n_on_and_activated)
                    self.assertEqual(1, er_stats.n_on_and_not_activated)
                    self.assertEqual(0, er_stats.n_off_and_activated)
                    self.assertEqual(0, er_stats.n_off_and_not_activated)
                elif i in lost:
                    self.assertEqual(0, er_stats.n_on_and_activated)
                    self.assertEqual(0, er_stats.n_on_and_not_activated)
                    self.assertEqual(0, er_stats.n_off_and_activated)
                    self.assertEqual(1, er_stats.n_off_and_not_activated)

    def test_reliability_0(self):
        # reliability before update should be NAN
        self.assertIs(np.NAN, self.schema.reliability)

    def test_reliability_1(self):
        # success update
        update_schema(self.schema,
                      activated=True,
                      succeeded=True,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,4,5'),
                      count=1)
        self.assertEqual(1.0, self.schema.reliability)

        # failure update
        update_schema(self.schema,
                      activated=True,
                      succeeded=False,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,5'),
                      count=1)
        self.assertEqual(0.5, self.schema.reliability)

        # failure update
        update_schema(self.schema,
                      activated=True,
                      succeeded=False,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,5'),
                      count=2)
        self.assertEqual(0.25, self.schema.reliability)

    def test_reliability_2(self):
        # failure update
        update_schema(self.schema,
                      activated=True,
                      succeeded=False,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,5'),
                      count=1)
        self.assertEqual(0.0, self.schema.reliability)

        # success update
        update_schema(self.schema,
                      activated=True,
                      succeeded=True,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,4,5'),
                      count=1)
        self.assertEqual(0.5, self.schema.reliability)

        # success update
        update_schema(self.schema,
                      activated=True,
                      succeeded=True,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,4,5'),
                      count=2)
        self.assertEqual(0.75, self.schema.reliability)

    def test_reliability_3(self):
        # reliability stats SHOULD NOT affected when schema not activated

        # failure update WITHOUT activation
        update_schema(self.schema,
                      activated=False,
                      succeeded=False,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,5'),
                      count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITHOUT activation
        update_schema(self.schema,
                      activated=False,
                      succeeded=True,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,4,5'),
                      count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # failure update WITHOUT activation
        update_schema(self.schema,
                      activated=False,
                      succeeded=False,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,5'),
                      count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITHOUT activation
        update_schema(self.schema,
                      activated=False,
                      succeeded=True,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,4,5'),
                      count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITH activation
        update_schema(self.schema,
                      activated=True,
                      succeeded=True,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,4,5'),
                      count=1)
        self.assertEqual(1.0, self.schema.reliability)

        # failure update WITHOUT activation
        update_schema(self.schema,
                      activated=False,
                      succeeded=False,
                      s_prev=sym_state('0,1'),
                      s_curr=sym_state('2,3,5'),
                      count=1)
        self.assertEqual(1.0, self.schema.reliability)

    def test_receive_from_extended_result(self):
        # only bare schemas have an extended result
        schema = sym_schema('/A1/')
        schema.notify_all = MagicMock()

        new_relevant_items = frozenset([sym_item('1'), sym_item('2')])
        schema.extended_result.new_relevant_items = new_relevant_items
        schema.receive(
            source=schema.extended_result,
            spin_off_type=SchemaSpinOffType.RESULT,
            another_kwarg='another_kwarg_value'
        )

        schema.notify_all.assert_called_once_with(
            source=schema,
            spin_off_type=SchemaSpinOffType.RESULT,
            relevant_items=new_relevant_items,
            another_kwarg='another_kwarg_value'
        )

    def test_receive_from_extended_context(self):
        schema = sym_schema('/A1/1,2')
        schema.notify_all = MagicMock()

        new_relevant_items = frozenset([sym_item('3'), sym_item('4')])
        schema.extended_context.new_relevant_items = new_relevant_items
        schema.receive(
            source=schema.extended_context,
            spin_off_type=SchemaSpinOffType.CONTEXT,
            another_kwarg='another_kwarg_value'
        )

        schema.notify_all.assert_called_once_with(
            source=schema,
            spin_off_type=SchemaSpinOffType.CONTEXT,
            relevant_items=new_relevant_items,
            another_kwarg='another_kwarg_value'
        )

    def test_notify_all(self):
        params = get_global_params()
        params.set('ext_context.correlation_test', DrescherCorrelationTest())

        self.schema.notify_all = MagicMock()

        act_state = sym_state('0,1,8')
        result_state = sym_state('2,3,4,7')

        # activated + success
        update_schema(self.schema, activated=True, succeeded=True, s_prev=act_state, s_curr=result_state, count=1)

        act_state = sym_state('0,1')
        result_state = sym_state('2,3,7')

        # activated + not success
        update_schema(self.schema, activated=True, succeeded=False, s_prev=act_state, s_curr=result_state, count=1)
        self.schema.notify_all.assert_called()

        act_state = sym_state('0,1,7')
        result_state = sym_state('2')

        # not activated and not success
        update_schema(self.schema, activated=False, succeeded=False, s_prev=act_state, s_curr=result_state, count=1)
        self.schema.notify_all.assert_called()

    def test_eq(self):
        self.assertTrue(satisfies_equality_checks(
            obj=self.schema,
            other=sym_schema('1,2/A1/3,4,5'),
            other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.schema))

    def test_serialize_basic_schema(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-schema-serialize.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.schema, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered = deserialize(path)

            self.assertEqual(self.schema, recovered)

    def test_serialize_composite_action_schema(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-schema-serialize_composite_action_schema.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.schema_ca, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered: Schema = deserialize(path)

            self.assertEqual(self.schema_ca, recovered)

    @test_share.performance_test
    def test_performance_1(self):
        n_items = 10_000
        n_state_elements = 25

        # populate item pool
        self.item_pool.clear()
        for i in range(n_items):
            self.item_pool.get(str(i))

        s_prev = tuple(map(str, sample(range(n_items), k=n_state_elements)))
        s_curr = tuple(map(str, sample(range(n_items), k=n_state_elements)))

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        start = time()
        self.schema.update(activated=True,
                           succeeded=True,
                           selection_state=s_prev,
                           new=new,
                           lost=lost)
        end = time()
        elapsed_time = end - start

        print(f'Time updating a schema with {n_items:,} items : {elapsed_time}s ')

    @test_share.performance_test
    def test_performance_2(self):
        # NOTE: n_items does not seem to impact performance much
        n_items = 100_000

        # NOTE: elapsed time seems to scale linearly with n_context_elements
        n_context_elements = 3

        # NOTE: elapsed time seems to scale linearly with n_schemas
        n_schemas = 10_000

        # populate item pool
        self.item_pool.clear()
        for i in range(n_items):
            self.item_pool.get(str(i))

        schemas = [
            Schema(context=sym_state_assert(','.join(str(i) for i in sample(range(n_items), k=n_context_elements))),
                   action=Action()) for _ in range(n_schemas)
        ]

        state = tuple(sample(range(n_items), k=n_context_elements))

        start = time()
        for s in schemas:
            s.is_applicable(state)
        end = time()
        elapsed_time = end - start

        print(f'Time determining if {n_schemas:,} schemas are applicable: {elapsed_time}s ')

    @test_share.performance_test
    def test_performance_3(self):
        n_iterations = 100_000

        other = Schema(
            context=sym_state_assert('1,2'),
            action=Action(),
            result=sym_state_assert('3,4,5'),
        )

        start = time()
        for _ in range(n_iterations):
            _ = self.schema == other
        end = time()
        elapsed_time = end - start

        print(f'Time for {n_iterations:,} calls to Schema.__eq__ comparing unequal objects: {elapsed_time}s ')

        start = time()
        for _ in range(n_iterations):
            _ = self.schema == copy(self.schema)
        end = time()
        elapsed_time = end - start

        print(f'Time for {n_iterations:,} calls to Schema.__eq__ comparing equal objects: {elapsed_time}s ')

    @test_share.performance_test
    def test_performance_4(self):
        n_iterations = 100_000

        start = time()
        for _ in range(n_iterations):
            _ = hash(self.schema)
        end = time()
        elapsed_time = end - start

        print(f'Time for {n_iterations:,} calls to Schema.__hash__: {elapsed_time}s ')

    @test_share.string_test
    def test_str(self):
        schema = Schema(action=Action('A1'))
        print(schema)

        schema = Schema(action=CompositeAction(sym_state_assert('1,2,~3')))
        print(schema)


class TestSchemaFunctions(TestCase):
    def setUp(self):
        common_test_setup()

    # noinspection PyPropertyAccess
    def test_is_reliable(self):
        reliability_threshold = 0.5

        params = get_global_params()
        params.set('reliability_threshold', reliability_threshold)

        schema = sym_schema('1,2/A1/3,4', schema_type=MockSchema)

        # test: should return False for schemas with reliabilities < threshold
        for reliability in np.linspace(0.0, reliability_threshold, endpoint=False):
            schema.reliability = reliability
            self.assertEqual(reliability, schema.reliability)
            self.assertFalse(is_reliable(schema))

        # test: should return False for schemas with reliabilities < threshold
        for reliability in np.linspace(reliability_threshold, 1.0):
            schema.reliability = reliability
            self.assertEqual(reliability, schema.reliability)
            self.assertTrue(is_reliable(schema))
