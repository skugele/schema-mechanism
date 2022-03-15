import itertools
import unittest
from collections import defaultdict

import numpy as np

from schema_mechanism.core import Chain
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaTree
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.share import GlobalParams
from test_share.test_classes import MockSchema
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestShared(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # allows direct setting of reliability (only reliable schemas are eligible for chaining)
        GlobalParams().set('schema_type', MockSchema)

        # setting learning rate to 1.0 to simplify testing
        GlobalParams().set('learning_rate', 1.0)

        # construct a SchemaTree for testing proximity to goal states
        self.s1 = sym_schema('/A1/', reliability=0.0, avg_duration=np.inf)
        self.s2 = sym_schema('/A2/', reliability=0.0, avg_duration=np.inf)
        self.s3 = sym_schema('/A3/', reliability=0.0, avg_duration=np.inf)
        self.s4 = sym_schema('/A4/', reliability=0.0, avg_duration=np.inf)

        self.s1_b = sym_schema('/A1/B,', reliability=0.25, avg_duration=np.inf)
        self.s1_d = sym_schema('/A1/D,', reliability=0.25, avg_duration=np.inf)
        self.s2_c = sym_schema('/A2/C,', reliability=0.25, avg_duration=np.inf)
        self.s3_d = sym_schema('/A3/D,', reliability=0.25, avg_duration=np.inf)
        self.s3_e = sym_schema('/A3/E,', reliability=0.25, avg_duration=np.inf)
        self.s4_k = sym_schema('/A4/K,', reliability=0.25, avg_duration=np.inf)
        self.s4_l = sym_schema('/A4/L,', reliability=0.25, avg_duration=np.inf)

        self.s1_a_b = sym_schema('A,/A1/B,', reliability=1.0, avg_duration=1.0)

        self.s2_b_c = sym_schema('B,/A2/C,', reliability=1.0, avg_duration=4.5)
        self.s3_b_c = sym_schema('B,/A3/C,', reliability=1.0, avg_duration=0.5)

        self.s1_c_d = sym_schema('C,/A1/D,', reliability=1.0, avg_duration=0.5)
        self.s3_c_d = sym_schema('C,/A3/D,', reliability=1.0, avg_duration=3.0)

        self.s4_d_k = sym_schema('D,/A4/K,', reliability=1.0, avg_duration=1.0)
        self.s4_k_l = sym_schema('K,/A4/L,', reliability=1.0, avg_duration=1.0)
        self.s3_l_e = sym_schema('L,/A3/E,', reliability=1.0, avg_duration=3.0)

        self.s1_b_e = sym_schema('D,/A1/E,', reliability=1.0, avg_duration=10.0)
        self.s3_d_e = sym_schema('D,/A3/E,', reliability=1.0, avg_duration=4.5)

        self.tree = SchemaTree(primitives=[self.s1, self.s2, self.s3, self.s4])

        self.tree.add_result_spin_offs(self.s1, [self.s1_b])
        self.tree.add_result_spin_offs(self.s2, [self.s2_c])
        self.tree.add_result_spin_offs(self.s3, [self.s3_d, self.s3_e])
        self.tree.add_result_spin_offs(self.s4, [self.s4_l, self.s4_k])

        self.tree.add_context_spin_offs(self.s1_b, [self.s1_a_b])
        self.tree.add_context_spin_offs(self.s1_d, [self.s1_c_d])
        self.tree.add_context_spin_offs(self.s2_c, [self.s2_b_c])
        self.tree.add_context_spin_offs(self.s3_d, [self.s3_c_d])
        self.tree.add_context_spin_offs(self.s3_e, [self.s3_l_e])

        self.tree.add_context_spin_offs(self.s4_k, [self.s4_d_k])
        self.tree.add_context_spin_offs(self.s4_l, [self.s4_k_l])

        self.sm = SchemaMemory.from_tree(self.tree)

        self.goal_state = sym_state_assert('E')
        self.controller = CompositeAction.Controller(self.goal_state)


class TestCompositeAction(TestShared):
    def setUp(self) -> None:
        super().setUp()

        self.goal_state = sym_state_assert('E')
        self.label = 'CA1'

        self.ca = CompositeAction(goal_state=self.goal_state, label=self.label)
        self.other = CompositeAction(goal_state=sym_state_assert('F'), label='other')

    # noinspection PyTypeChecker,PyArgumentList
    def test_init(self):
        # test: the goal state should be required and set properly
        self.assertRaises(TypeError, lambda: CompositeAction())
        self.assertRaises(ValueError, lambda: CompositeAction(goal_state=None))

        # test: goal state should be properly initialized
        self.assertEqual(self.goal_state, self.ca.goal_state)

        # test: controller should be properly initialized
        self.assertIsNotNone(self.ca.controller)

        # test: an optional label should be set if provided
        self.assertEqual(self.label, self.ca.label)
        self.assertIsNone(CompositeAction(self.goal_state).label)

        # test: a unique id should be assigned
        self.assertIsNotNone(self.ca.uid)

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

    def test_equals(self):
        self.assertTrue(satisfies_equality_checks(obj=self.ca, other=self.other))

        # test: goal state is used to determine equality NOT labels
        ca_1 = CompositeAction(sym_state_assert('1,2'), label='CA1')
        ca_2 = CompositeAction(sym_state_assert('1,2'), label='CA2')
        ca_3 = CompositeAction(sym_state_assert('1,3'), label='CA1')

        self.assertEqual(ca_1, ca_2)
        self.assertNotEqual(ca_1, ca_3)
        self.assertNotEqual(ca_2, ca_3)

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.ca))


class TestController(TestShared):

    # TODO: This replicates the majority of the setup from TestCompositeAction. Need to move the common
    # TODO: code into a shared function.
    def setUp(self) -> None:
        super().setUp()

        self.goal_state = sym_state_assert('E')
        self.controller = CompositeAction.Controller(self.goal_state)

    def test_init(self):
        # test: components should be an empty set
        self.assertSetEqual(set(), set(self.controller.components))

    def test_update_empty_list_of_chains(self):
        # test: empty list of chains should result in empty components
        self.controller.update(chains=list())
        self.assertSetEqual(set(), set(self.controller.components))

    def test_update_with_single_link_chain(self):
        chain = [Chain([self.s3_d_e])]
        self.controller.update(chain)

        # test: single link chain should result in a single component schema
        component = chain[0][0]
        self.assertSetEqual({component}, set(self.controller.components))

        # test: goal proximity of component schema should be equal to the reciprocal of its average duration
        self.assertEqual(1.0 / component.avg_duration, self.controller.proximity(component))

    def test_update_with_multi_link_chain(self):
        chains = [Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e])]
        self.controller.update(chains)

        # test: multiple link chain should result in a component schema for each link in the chain
        components = list(itertools.chain.from_iterable(c for c in chains))
        self.assertSetEqual(set(components), set(self.controller.components))

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
        self.assertSetEqual(set(components), self.controller.components)

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

        expected_proximities = defaultdict(lambda: np.inf)
        for chain in chains:
            for schema in chain:
                expected_proximities[schema] = np.min(
                    [expected_proximities[schema], self._calc_proximity(schema, chain)])

        # test: a component's goal proximity should be its min proximity to goal if the schema appears in multiple chains
        actual_proximities = {s: self.controller.proximity(s) for s in self.controller.components}
        for component in components:
            self.assertEqual(expected_proximities[component], actual_proximities[component])

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

        expected_proximities = defaultdict(lambda: np.inf)
        for chain in chains:
            for schema in chain:
                expected_proximities[schema] = np.min(
                    [expected_proximities[schema], self._calc_proximity(schema, chain)])

        # test: a component's proximity should be the reciprocal of the sum of avg durations in the remaining chain
        actual_proximities = {s: self.controller.proximity(s) for s in self.controller.components}
        for component in components:
            self.assertEqual(expected_proximities[component], actual_proximities[component])

    def _calc_proximity(self, schema: Schema, chain: Chain) -> float:
        start = chain.index(schema)

        # this calculation assumes a learning rate of 1.0
        return 1.0 / sum(s.avg_duration for s in itertools.islice(chain, start, None))
