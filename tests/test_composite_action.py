import itertools
import unittest
from collections import defaultdict
from typing import Any

import numpy as np

from schema_mechanism.core import Chain
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import Controller
from schema_mechanism.core import ControllerMap
from schema_mechanism.core import DummyController
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import get_controller_map
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import sym_assert
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.serialization.json.decoders import decode
from schema_mechanism.serialization.json.encoders import encode
from test_share import disable_test
from test_share.test_classes import MockSchema
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestShared(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # setting learning rate to 1.0 to simplify testing
        get_global_params().set('learning_rate', 1.0)

        # construct a SchemaTree for testing proximity to goal states
        self.s1 = sym_schema('/A1/', schema_type=MockSchema, reliability=0.0, avg_duration=np.inf)
        self.s2 = sym_schema('/A2/', schema_type=MockSchema, reliability=0.0, avg_duration=np.inf)
        self.s3 = sym_schema('/A3/', schema_type=MockSchema, reliability=0.0, avg_duration=np.inf)
        self.s4 = sym_schema('/A4/', schema_type=MockSchema, reliability=0.0, avg_duration=np.inf)

        self.s1_b = sym_schema('/A1/B,', schema_type=MockSchema, reliability=0.25, avg_duration=np.inf)
        self.s1_d = sym_schema('/A1/D,', schema_type=MockSchema, reliability=0.25, avg_duration=np.inf)
        self.s2_c = sym_schema('/A2/C,', schema_type=MockSchema, reliability=0.25, avg_duration=np.inf)
        self.s3_d = sym_schema('/A3/D,', schema_type=MockSchema, reliability=0.25, avg_duration=np.inf)
        self.s3_e = sym_schema('/A3/E,', schema_type=MockSchema, reliability=0.25, avg_duration=np.inf)
        self.s4_k = sym_schema('/A4/K,', schema_type=MockSchema, reliability=0.25, avg_duration=np.inf)
        self.s4_l = sym_schema('/A4/L,', schema_type=MockSchema, reliability=0.25, avg_duration=np.inf)

        self.s1_a_b = sym_schema('A,/A1/B,', schema_type=MockSchema, reliability=1.0, avg_duration=1.0)

        self.s2_b_c = sym_schema('B,/A2/C,', schema_type=MockSchema, reliability=1.0, avg_duration=4.5)
        self.s3_b_c = sym_schema('B,/A3/C,', schema_type=MockSchema, reliability=1.0, avg_duration=0.5)

        self.s1_c_d = sym_schema('C,/A1/D,', schema_type=MockSchema, reliability=1.0, avg_duration=0.5)
        self.s3_c_d = sym_schema('C,/A3/D,', schema_type=MockSchema, reliability=1.0, avg_duration=3.0)

        self.s4_d_k = sym_schema('D,/A4/K,', schema_type=MockSchema, reliability=1.0, avg_duration=1.0)
        self.s4_k_l = sym_schema('K,/A4/L,', schema_type=MockSchema, reliability=1.0, avg_duration=1.0)
        self.s3_l_e = sym_schema('L,/A3/E,', schema_type=MockSchema, reliability=1.0, avg_duration=3.0)

        self.s1_b_e = sym_schema('D,/A1/E,', schema_type=MockSchema, reliability=1.0, avg_duration=10.0)
        self.s3_d_e = sym_schema('D,/A3/E,', schema_type=MockSchema, reliability=1.0, avg_duration=4.5)

        self.tree = SchemaTree()
        self.tree.add(schemas=[self.s1, self.s2, self.s3, self.s4])

        self.tree.add(source=self.s1, schemas=[self.s1_b])
        self.tree.add(source=self.s2, schemas=[self.s2_c])
        self.tree.add(source=self.s3, schemas=[self.s3_d, self.s3_e])
        self.tree.add(source=self.s4, schemas=[self.s4_l, self.s4_k])

        self.tree.add(source=self.s1_b, schemas=[self.s1_a_b])
        self.tree.add(source=self.s1_d, schemas=[self.s1_c_d])
        self.tree.add(source=self.s2_c, schemas=[self.s2_b_c])
        self.tree.add(source=self.s3_d, schemas=[self.s3_c_d])
        self.tree.add(source=self.s3_e, schemas=[self.s3_l_e])

        self.tree.add(source=self.s4_k, schemas=[self.s4_d_k])
        self.tree.add(source=self.s4_l, schemas=[self.s4_k_l])

        self.sm = SchemaMemory(self.tree)

        self.goal_state = sym_state_assert('E')
        self.controller = Controller(self.goal_state)


class TestCompositeAction(TestShared):
    def setUp(self) -> None:
        super().setUp()

        self.goal_state = sym_state_assert('E')
        self.label = 'CA1'

        self.composite_action = CompositeAction(goal_state=self.goal_state, label=self.label)
        self.other = CompositeAction(goal_state=sym_state_assert('F'), label='other')

    # noinspection PyTypeChecker,PyArgumentList
    def test_init(self):
        # test: the goal state should be required and set properly
        self.assertRaises(TypeError, lambda: CompositeAction())
        self.assertRaises(ValueError, lambda: CompositeAction(goal_state=None))

        # test: goal state should be properly initialized
        self.assertEqual(self.goal_state, self.composite_action.goal_state)

        # test: controller should be properly initialized
        self.assertIsNotNone(self.composite_action.controller)

        # test: an optional label should be set if provided
        self.assertEqual(self.label, self.composite_action.label)
        self.assertIsNone(CompositeAction(self.goal_state).label)

        # test: a unique id should be assigned
        self.assertIsNotNone(self.composite_action.uid)

    def test_is_enabled(self):
        ca = CompositeAction(goal_state=self.goal_state)

        # sanity checks: verifies CompositeAction has no components
        self.assertEqual(0, len(ca.controller.components))

        # test: composite action with no composite schemas SHOULD NOT be enabled in ANY state
        self.assertFalse(ca.is_enabled(sym_state('A')))
        self.assertFalse(ca.is_enabled(sym_state('B')))
        self.assertFalse(ca.is_enabled(sym_state('A,B,C,D,E,F,G')))

        # adds components for next test case
        chains = [Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e])]
        ca.controller.update(chains)

        # sanity check: verifies CompositeAction has expected components needed for test case
        self.assertSetEqual(set(chains[0]), ca.controller.components)

        # test: composite action with applicable component (context matches State) SHOULD be enabled
        for state in {sym_state(se) for se in 'ABCD'}:
            self.assertTrue(ca.is_enabled(state))

        # test: composite action without applicable components (context does not matches State) SHOULD NOT be enabled
        for state in {sym_state(se) for se in 'EJ1'}:
            self.assertFalse(ca.is_enabled(state))

    def test_update_basic_tests_1(self):
        # test: update with an empty chain should be allowed and leave components unchanged
        ca = CompositeAction(goal_state=sym_state_assert('S'))

        self.assertSetEqual(set(), ca.controller.components)
        ca.controller.update([])
        self.assertSetEqual(set(), ca.controller.components)

        ca.controller.update([Chain()])
        self.assertSetEqual(set(), ca.controller.components)

    def test_update_basic_tests_2(self):
        # test: update with single chain that does not end in goal state should raise a ValueError
        ca = CompositeAction(goal_state=sym_state_assert('S'))

        self.assertRaises(ValueError, lambda: ca.controller.update([Chain([sym_schema('/A,/1,')])]))

    # FIXME: Enable this test case if/when controller components with composite actions are supported
    @disable_test
    def test_update_basic_tests_3(self):
        # test: chains that contain schemas with the same composite action as the controller should be ignored
        ca = CompositeAction(goal_state=sym_state_assert('S'))

        ca.controller.update([Chain([sym_schema('/S,/S,')])])
        self.assertSetEqual(set(), ca.controller.components)

    # FIXME: Enable this test case if/when controller components with composite actions are supported
    @disable_test
    def test_update_basic_tests_4(self):
        # test: chains that contain schemas with the same composite action as the controller should be ignored
        ca = CompositeAction(goal_state=sym_state_assert('S2'))

        schema_ca = sym_schema('A,/D,/M,')
        chains = [
            Chain([sym_schema('A,/S2,/B,'), self.s2_b_c, self.s3_c_d]),
        ]
        schema_ca.action.controller.update(chains)

        schema_term = sym_schema('M,/A1/S2,')
        ca.controller.update([Chain([schema_ca, schema_term])])

        # test: only schema_term should be added (part of chain that occurs after the recursive schema)
        self.assertSetEqual({schema_term}, ca.controller.components)

    # FIXME: Enable this test case if/when controller components with composite actions are supported
    @disable_test
    def test_update_component_with_controller_containing_parent_action(self):
        # 1st composite action schema
        s1 = sym_schema('M1,P/M1,/P,')

        # s1's components
        s1_c1 = sym_schema('/sit[M1]/M1,', schema_type=MockSchema, reliability=1.0)
        s1_c2 = sym_schema('/M1,P/M1,P', schema_type=MockSchema, reliability=1.0)
        s1_c3 = sym_schema('S,/sit[M1]/M1,', schema_type=MockSchema, reliability=1.0)
        s1_c4 = sym_schema('/stand/S,', schema_type=MockSchema, reliability=1.0)

        chains = [
            Chain([s1_c1]),
            Chain([s1_c2]),
            Chain([s1_c4, s1_c3]),
        ]
        s1.action.controller.update(chains)

        self.assertSetEqual({s1_c1, s1_c3, s1_c4}, set(s1.action.controller.components))

        # 2nd composite action schema
        s2 = sym_schema('/M1,P/M1,P')

        # s2's components
        s2_c1 = sym_schema('/stand/S,', schema_type=MockSchema, reliability=1.0)
        s2_c2 = sym_schema('M1,/P,/M1,P', schema_type=MockSchema, reliability=1.0)
        s2_c3 = sym_schema('/sit[M1]/M1,', schema_type=MockSchema, reliability=1.0)
        s2_c4 = sym_schema('S,/sit[M1]/M1,', schema_type=MockSchema, reliability=1.0)
        s2_c5 = sym_schema('M1,/deposit/M1,P', schema_type=MockSchema, reliability=1.0)

        chains = [
            Chain([s2_c3, s2_c5]),
            Chain([s2_c3, s2_c2]),
            Chain([s2_c1, s2_c4, s2_c5]),
        ]
        s2.action.controller.update(chains)

        # 3rd composite action schema
        s3 = sym_schema('M1,/P,/M1,P')

        # s3's components
        s3_c1 = sym_schema('M1,P/M1,/P,', schema_type=MockSchema, reliability=1.0)

        chains = [
            Chain([s3_c1]),
        ]
        s3.action.controller.update(chains)

        s3.is_applicable(sym_state('M1,P'))

    def test_equals(self):
        self.assertTrue(
            satisfies_equality_checks(obj=self.composite_action, other_same_type=self.other, other_different_type=1.0))

        # test: goal state is used to determine equality NOT labels
        ca_1 = CompositeAction(sym_state_assert('1,2'), label='CA1')
        ca_2 = CompositeAction(sym_state_assert('1,2'), label='CA2')
        ca_3 = CompositeAction(sym_state_assert('1,3'), label='CA1')

        self.assertEqual(ca_1, ca_2)
        self.assertNotEqual(ca_1, ca_3)
        self.assertNotEqual(ca_2, ca_3)

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.composite_action))

    def test_controller_sharing(self):
        # test: composite actions with the same goal state should share the same controller

        # test: all of these should have the same controller instance
        ca1 = CompositeAction(sym_state_assert('1,2'), label='first')
        ca2 = CompositeAction(sym_state_assert('1,2'), label='second')
        ca3 = CompositeAction(sym_state_assert('1,2'), label='third')

        self.assertIs(ca1.controller, ca2.controller)
        self.assertIs(ca2.controller, ca3.controller)

        # test: these composite actions should have different controller instances
        ca4 = CompositeAction(sym_state_assert('1,3'), label='another')
        ca5 = CompositeAction(sym_state_assert('1,4,5'), label='yet another')

        self.assertIsNot(ca1.controller, ca4.controller)
        self.assertIsNot(ca1.controller, ca5.controller)

    def test_encode_and_decode(self):
        object_registry: dict[int, Any] = dict()
        encoded_obj = encode(self.composite_action, object_registry=object_registry)
        decoded_obj: CompositeAction = decode(encoded_obj, object_registry=object_registry)

        self.assertEqual(self.composite_action, decoded_obj)
        self.assertEqual(self.goal_state, decoded_obj.goal_state)
        self.assertEqual(self.label, decoded_obj.label)


class TestController(TestShared):
    def setUp(self) -> None:
        super().setUp()

        self.goal_state = sym_state_assert('E')
        self.controller = Controller(self.goal_state)

    def test_init(self):
        # test: components should be an empty set
        self.assertSetEqual(set(), set(self.controller.components))
        self.assertSetEqual(set(), set(self.controller.descendants))

    def test_update_empty_list_of_chains(self):
        # test: empty list of chains should result in empty components
        self.controller.update(chains=list())
        self.assertSetEqual(set(), set(self.controller.components))
        self.assertSetEqual(set(), set(self.controller.descendants))

    def test_update_with_single_link_chain(self):
        chain = [Chain([self.s3_d_e])]
        self.controller.update(chain)

        # test: single link chain should result in a single component schema
        component = chain[0][0]
        self.assertSetEqual({component}, set(self.controller.components))
        self.assertSetEqual({component}, set(self.controller.descendants))

        # test: goal proximity of component schema should be equal to the reciprocal of its average duration
        self.assertEqual(1.0 / component.avg_duration, self.controller.proximity(component))

    def test_update_with_multi_link_chain(self):
        chains = [Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e])]
        self.controller.update(chains)

        # test: multiple link chain should result in a component schema for each link in the chain
        components = list(itertools.chain.from_iterable(c for c in chains))
        self.assertSetEqual(set(components), set(self.controller.components))
        self.assertSetEqual(set(components), set(self.controller.descendants))

        # test: goal proximity of component schemas should be equal to the sum of the average duration of schemas in
        #       the chain from it to the goal state
        proximities = {
            self.s1_a_b: self._calc_proximity(self.s1_a_b, chains[0]),
            self.s2_b_c: self._calc_proximity(self.s2_b_c, chains[0]),
            self.s3_c_d: self._calc_proximity(self.s3_c_d, chains[0]),
            self.s3_d_e: self._calc_proximity(self.s3_d_e, chains[0]),
        }

        for component in components:
            self.assertEqual(proximities[component], self.controller.proximity(component))

    def test_update_with_multiple_distinct_chains(self):
        chains = [Chain([self.s1_a_b, self.s1_b_e]),
                  Chain([self.s3_c_d, self.s3_d_e])]
        self.controller.update(chains)

        # test: should be a component schema for each schema in each chain
        components = list(itertools.chain.from_iterable(c for c in chains))
        self.assertSetEqual(set(components), set(self.controller.components))
        self.assertSetEqual(set(components), set(self.controller.descendants))

        proximities = {
            self.s1_a_b: self._calc_proximity(self.s1_a_b, chains[0]),
            self.s1_b_e: self._calc_proximity(self.s1_b_e, chains[0]),
            self.s3_c_d: self._calc_proximity(self.s3_c_d, chains[1]),
            self.s3_d_e: self._calc_proximity(self.s3_d_e, chains[1]),
        }

        # test: goal proximity of component schemas should be equal to the sum of the average duration of schemas in
        #       the chain from it to the goal state
        for component in components:
            self.assertEqual(proximities[component], self.controller.proximity(component))

    def test_update_with_multiple_overlapping_chains(self):
        chains = [
            Chain([self.s1_a_b, self.s3_b_c, self.s1_c_d, self.s4_d_k, self.s4_k_l, self.s3_l_e]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
            Chain([self.s3_c_d, self.s3_d_e]),
            Chain([self.s1_b_e]),
        ]
        self.controller.update(chains)

        # test: should be a component schema for each schema in each chain
        components = list(itertools.chain.from_iterable(c for c in chains))
        self.assertSetEqual(set(components), self.controller.components)
        self.assertSetEqual(set(components), set(self.controller.descendants))

        expected_proximities = defaultdict(lambda: np.inf)
        for chain in chains:
            for schema in chain:
                expected_proximities[schema] = np.min(
                    [expected_proximities[schema], self._calc_proximity(schema, chain)])

        # test: component goal proximity should be min proximity to goal if the schema appears in multiple chains
        actual_proximities = {s: self.controller.proximity(s) for s in self.controller.components}
        for component in components:
            self.assertEqual(expected_proximities[component], actual_proximities[component])

    def test_chain_with_composite_action_schema(self):
        composite_action_schema_a_b = sym_schema('A,/G1,/B,')
        composite_action_schema_b_c = sym_schema('B,/G1,/C,')
        composite_action_schema_c_d = sym_schema('C,/G1,/D,')
        composite_action_schema_d_e = sym_schema('D,/G1,/E,')

        chains = [
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, composite_action_schema_d_e]),
            Chain([self.s1_a_b, self.s2_b_c, composite_action_schema_c_d, self.s3_d_e]),
            Chain([self.s1_a_b, composite_action_schema_b_c, self.s3_c_d, self.s3_d_e]),
            Chain([composite_action_schema_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
        ]

        for i, chain in enumerate(chains):
            controller = Controller(self.goal_state)
            controller.update([chain])

            # test: the chain should be broken when a composite action schema is encountered
            expected_components = list(reversed(chain))[:i]

            self.assertSetEqual(set(expected_components), controller.components)

    def test_all_satisfied_by(self):
        # clears all of the registered controllers
        controller_map: ControllerMap = get_controller_map()
        controller_map.clear()

        controllers: dict = {
            'A': CompositeAction(goal_state=sym_assert('A')).controller,
            'A,B,C': CompositeAction(goal_state=sym_assert('A,B,C')).controller,
            'X': CompositeAction(goal_state=sym_assert('X')).controller,
            'X,Y,Z': CompositeAction(goal_state=sym_assert('X,Y,Z')).controller,
        }

        all_controllers: set = {controller for controller in controllers.values()}

        # test: no controllers should be satisfied by these goal state
        self.assertSetEqual(set(), CompositeAction.all_satisfied_by(sym_state('1')))
        self.assertSetEqual(set(), CompositeAction.all_satisfied_by(sym_state('101')))

        # test: all controllers should be satisfied by these goal state
        self.assertSetEqual(all_controllers, CompositeAction.all_satisfied_by(state=sym_state('A,B,C,X,Y,Z')))
        self.assertSetEqual(all_controllers, CompositeAction.all_satisfied_by(state=sym_state('A,B,C,X,Y,Z,1,2,3')))

        # test: these states should match some, but not all, of the controllers
        self.assertSetEqual({controllers['A']}, CompositeAction.all_satisfied_by(state=sym_state('A')))
        self.assertSetEqual({controllers['A']}, CompositeAction.all_satisfied_by(state=sym_state('A,B')))
        self.assertSetEqual({controllers['A'], controllers['A,B,C']},
                            CompositeAction.all_satisfied_by(state=sym_state('A,B,C')))
        self.assertSetEqual({controllers['X']}, CompositeAction.all_satisfied_by(state=sym_state('X')))
        self.assertSetEqual({controllers['X']}, CompositeAction.all_satisfied_by(state=sym_state('X,Y')))
        self.assertSetEqual({controllers['X'], controllers['X,Y,Z']},
                            CompositeAction.all_satisfied_by(state=sym_state('X,Y,Z')))

    # FIXME: the behavior was changed to replace rather than update existing components. It may be desirable to
    # FIXME: to revert this based on additional experimentation. I am commenting out these test cases for now.
    @disable_test
    def test_multiple_updates_with_overlapping_chains(self):
        # test: same goal proximities given same chains whether single or multiple update invocations
        chains = [
            Chain([self.s1_a_b, self.s3_b_c, self.s1_c_d, self.s4_d_k, self.s4_k_l, self.s3_l_e]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
            Chain([self.s3_c_d, self.s3_d_e]),
            Chain([self.s1_b_e]),
        ]

        for chain in chains:
            self.controller.update([chain])

        # test: should be a component schema for each schema in each chain
        components = list(itertools.chain.from_iterable(c for c in chains))
        self.assertSetEqual(set(components), self.controller.components)
        self.assertSetEqual(set(components), set(self.controller.descendants))

        expected_proximities = defaultdict(lambda: np.inf)
        for chain in chains:
            for schema in chain:
                expected_proximities[schema] = np.min(
                    [expected_proximities[schema], self._calc_proximity(schema, chain)])

        # test: a component's proximity should be the reciprocal of the sum of avg durations in the remaining chain
        actual_proximities = {s: self.controller.proximity(s) for s in self.controller.components}
        for component in components:
            self.assertEqual(expected_proximities[component], actual_proximities[component])

    # FIXME: Enable this test case if/when controller components with composite actions are supported
    @disable_test
    def test_update_with_composite_action_component(self):
        schema_ca = sym_schema('A,/D,/M,')
        chains = [
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d]),
            Chain([self.s1_a_b, self.s2_b_c, self.s1_c_d]),
        ]
        schema_ca.action.controller.update(chains)

        chains = [Chain([schema_ca, self.s3_d_e])]
        self.controller.update(chains)

        # test: single link chain should result in a single component schema
        self.assertSetEqual({*chains[0]}, set(self.controller.components))
        self.assertSetEqual({*chains[0]}.union(schema_ca.action.controller.descendants),
                            set(self.controller.descendants))

        # test: a component's proximity should be the reciprocal of the sum of avg durations in the remaining chain
        expected_proximities = defaultdict(lambda: np.inf)
        for chain in chains:
            for schema in chain:
                expected_proximities[schema] = np.min(
                    [expected_proximities[schema], self._calc_proximity(schema, chain)])

        actual_proximities = {s: self.controller.proximity(s) for s in self.controller.components}
        for component in self.controller.components:
            self.assertEqual(expected_proximities[component], actual_proximities[component])

    # FIXME: Enable this test case if/when controller components with composite actions are supported
    # noinspection PyUnresolvedReferences
    @disable_test
    def test_contained_in(self):
        controller = Controller(goal_state=sym_state_assert('S2,'))

        self.assertFalse(controller.contained_in(sym_schema('A,/A1/B,')))
        self.assertFalse(controller.contained_in(sym_schema('A,/S1,/B,')))

        schema_ca = sym_schema('A,/D,/M,')
        chains = [
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d]),
            Chain([self.s1_a_b, self.s2_b_c, self.s1_c_d]),
        ]
        schema_ca.action.controller.update(chains)

        self.assertFalse(controller.contained_in(schema_ca))

        schema_ca = sym_schema('A,/D,/M,')
        schema_ca.action.controller.update([Chain([sym_schema('1,/S2,/2,'), sym_schema('2,/A2/D,')])])

        self.assertTrue(controller.contained_in(sym_schema('1,/S2,/2,')))
        self.assertTrue(controller.contained_in(schema_ca))

    def test_eq(self):
        # test: equal should function properly given controllers no components
        controller_1 = Controller(goal_state=sym_state_assert('D'))
        controller_2 = Controller(goal_state=sym_state_assert('E'))

        self.assertTrue(
            satisfies_equality_checks(obj=controller_1, other_same_type=controller_2, other_different_type=1.0))

        # test: equal should function properly given controllers with components
        controller_1 = Controller(goal_state=sym_state_assert('D'))
        controller_2 = Controller(goal_state=sym_state_assert('E'))

        controller_1.update([Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d])])
        controller_2.update([Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e])])

        self.assertTrue(
            satisfies_equality_checks(obj=controller_1, other_same_type=controller_2, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.controller))

    def test_encode_and_decode(self):

        controller = Controller(
            goal_state=sym_state_assert('4'),
            proximity_map={
                sym_schema('1,/A1/2,'): 1.0 / 3.0,
                sym_schema('2,/A2/3,'): 1.0 / 2.0,
                sym_schema('3,/A3/4,'): 1.0,
            },
            total_cost_map={
                sym_schema('1,/A1/2,'): 1.0,
                sym_schema('2,/A2/3,'): 2.0,
                sym_schema('3,/A3/4,'): 3.0,
            },
            components={
                sym_schema('1,/A1/2,'),
                sym_schema('2,/A2/3,'),
                sym_schema('3,/A3/4,'),
            },
        )

        object_registry = dict()
        encoded_obj = encode(controller, object_registry=object_registry)
        decoded_obj: Controller = decode(encoded_obj, object_registry=object_registry)

        self.assertEqual(controller, decoded_obj)
        self.assertEqual(controller.goal_state, decoded_obj.goal_state)
        self.assertDictEqual(controller.proximity_map, decoded_obj.proximity_map)
        self.assertDictEqual(controller.total_cost_map, decoded_obj.total_cost_map)

    def _calc_proximity(self, schema: Schema, chain: Chain) -> float:
        start = chain.index(schema)

        # this calculation assumes a learning rate of 1.0
        return 1.0 / sum(s.avg_duration for s in itertools.islice(chain, start, None))


class TestDummyController(TestShared):
    def setUp(self) -> None:
        super().setUp()

        self.dummy_controller = DummyController()

    def test_init(self):
        self.assertSetEqual(set(), self.dummy_controller.components)
        self.assertSetEqual(set(), self.dummy_controller.descendants)
        self.assertEqual(NULL_STATE_ASSERT, self.dummy_controller.goal_state)

    def test_update(self):
        self.assertRaises(NotImplementedError, lambda: self.dummy_controller.update())

    def test_proximity(self):
        for schema in self.sm:
            self.assertEqual(-np.inf, self.dummy_controller.proximity(schema))
