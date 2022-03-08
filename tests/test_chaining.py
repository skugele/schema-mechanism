import unittest

from schema_mechanism.core import GlobalParams
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import SchemaTree
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import Chain
from schema_mechanism.modules import SchemaMemory
from test_share.test_classes import MockSchema
from test_share.test_func import common_test_setup


class TestBackwardChaining(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # allows direct setting of reliability (only reliable schemas are eligible for chaining)
        GlobalParams().set('schema_type', MockSchema)

        self.s1 = sym_schema('/A1/', reliability=0.0)
        self.s2 = sym_schema('/A2/', reliability=0.0)
        self.s3 = sym_schema('/A3/', reliability=0.0)

        self.s1_b = sym_schema('/A1/B,', reliability=0.25)
        self.s2_c = sym_schema('/A2/C,', reliability=0.25)
        self.s3_d = sym_schema('/A3/D,', reliability=0.25)
        self.s3_e = sym_schema('/A3/E,', reliability=0.25)

        self.s1_a_b = sym_schema('A,/A1/B,', reliability=1.0)
        self.s2_b_c = sym_schema('B,/A2/C,', reliability=1.0)
        self.s3_c_d = sym_schema('C,/A3/D,', reliability=1.0)
        self.s3_d_e = sym_schema('D,/A3/E,', reliability=1.0)

        self.tree = SchemaTree(primitives=[self.s1, self.s2, self.s3])

        # L2 result spin-offs
        self.tree.add_result_spin_offs(self.s1, [self.s1_b])
        self.tree.add_result_spin_offs(self.s2, [self.s2_c])
        self.tree.add_result_spin_offs(self.s3, [self.s3_d, self.s3_e])

        # L2 context spin-offs
        self.tree.add_context_spin_offs(self.s1_b, [self.s1_a_b])
        self.tree.add_context_spin_offs(self.s2_c, [self.s2_b_c])
        self.tree.add_context_spin_offs(self.s3_d, [self.s3_c_d])
        self.tree.add_context_spin_offs(self.s3_e, [self.s3_d_e])

        self.sm = SchemaMemory.from_tree(self.tree)

    def test_base_cases_for_recursion(self):
        # test: NULL goal state should return an empty list
        self.assertListEqual([], self.sm.backward_chains(goal_state=NULL_STATE_ASSERT))
        self.assertListEqual([], self.sm.backward_chains(goal_state=NULL_STATE_ASSERT, max_len=None))
        self.assertListEqual([], self.sm.backward_chains(goal_state=NULL_STATE_ASSERT, max_len=10))

        # test: non-positive max length should return empty list
        self.assertListEqual([], self.sm.backward_chains(goal_state=sym_state_assert('E,'), max_len=0))

    def test_single_chain(self):
        expected_chain = []
        chains = self.sm.backward_chains(goal_state=sym_state_assert('A,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s1_a_b])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('B,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s1_a_b, self.s2_b_c])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('C,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('D,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('E,'))
        self.assertListEqual(expected_chain, list(chains))

    def test_multi_chain(self):
        # adds a second reliable chain from C to D
        s2_d = sym_schema('/A2/D,', reliability=0.25)
        s2_c_d = sym_schema('C,/A2/D,', reliability=1.0)

        self.tree.add_result_spin_offs(self.s2, [s2_d])
        self.tree.add_context_spin_offs(s2_d, [s2_c_d])

        self.sm = SchemaMemory.from_tree(self.tree)

        expected_chain = []
        chains = self.sm.backward_chains(goal_state=sym_state_assert('A,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s1_a_b])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('B,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s1_a_b, self.s2_b_c])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('C,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chains = [
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d]),
            Chain([self.s1_a_b, self.s2_b_c, s2_c_d])
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('D,'))
        self.assertSetEqual(set(expected_chains), set(chains))

        expected_chains = [
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
            Chain([self.s1_a_b, self.s2_b_c, s2_c_d, self.s3_d_e])
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('E,'))
        self.assertSetEqual(set(expected_chains), set(chains))

    def test_with_loop_1(self):
        # adds a second reliable chain from C to D
        s2_a = sym_schema('/A2/A,', reliability=0.25)
        s2_e_a = sym_schema('E,/A2/A,', reliability=1.0)

        self.tree.add_result_spin_offs(self.s2, [s2_a])
        self.tree.add_context_spin_offs(s2_a, [s2_e_a])

        expected_chain = [Chain([self.s2_b_c, self.s3_c_d, self.s3_d_e, s2_e_a])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('A,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s3_c_d, self.s3_d_e, s2_e_a, self.s1_a_b])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('B,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s3_d_e, s2_e_a, self.s1_a_b, self.s2_b_c])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('C,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([s2_e_a, self.s1_a_b, self.s2_b_c, self.s3_c_d])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('D,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('E,'))
        self.assertListEqual(expected_chain, list(chains))

    def test_with_loop_2(self):
        # adds a second reliable chain from B to A creating a loop B -> A -> B
        s1_b = sym_schema('/A1/B,', reliability=0.25)
        s1_b_a = sym_schema('B,/A1/A,', reliability=1.0)

        self.tree.add_result_spin_offs(self.s1, [s1_b])
        self.tree.add_context_spin_offs(s1_b, [s1_b_a])

        expected_chain = [Chain([self.s1_a_b])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('B,'))
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('E,'))
        self.assertListEqual(expected_chain, list(chains))

    def test_chains_with_reliable_no_context_schema(self):
        s2_e = sym_schema('/A2/E,', reliability=1.0)
        self.tree.add_result_spin_offs(self.s2, [s2_e])

        expected_chains = [
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
            Chain([s2_e]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('E,'))
        self.assertSetEqual(set(expected_chains), set(chains))

    def test_chains_with_identical_context_and_goal_schema(self):
        s2_e = sym_schema('E,/A2/E,', reliability=1.0)

        self.tree.add_result_spin_offs(self.s2, [s2_e])

        expected_chains = [
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('E,'))
        self.assertSetEqual(set(expected_chains), set(chains))

    def test_chains_with_composite_items(self):
        s3_cd_e = sym_schema('C,D,/A3/E,', reliability=1.0)
        s1_cd = sym_schema('/A1/(C,D),', reliability=1.0)
        s1_b_cd = sym_schema('B,/A1/(C,D),', reliability=1.0)

        self.tree.add_context_spin_offs(self.s3_d_e, [s3_cd_e])
        self.tree.add_result_spin_offs(self.s1, [s1_cd])
        self.tree.add_context_spin_offs(s1_cd, [s1_b_cd])

        expected_chains = [
            Chain([s1_cd]),
            Chain([self.s1_a_b, s1_b_cd]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('C,D'))
        self.assertSetEqual(set(expected_chains), set(chains))

        expected_chains = [
            Chain([s1_cd, self.s3_d_e]),
            Chain([s1_cd, self.s3_c_d, self.s3_d_e]),
            Chain([self.s1_a_b, s1_b_cd, self.s3_c_d, self.s3_d_e]),
            Chain([self.s1_a_b, s1_b_cd, self.s3_d_e]),
            Chain([s1_cd, s3_cd_e]),
            Chain([self.s1_a_b, s1_b_cd, s3_cd_e]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('E,'))
        self.assertSetEqual(set(expected_chains), set(chains))

    def test_chains_with_max_length(self):
        expected_chain = [Chain([self.s2_b_c, self.s3_c_d])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('D,'), max_len=2)
        self.assertListEqual(expected_chain, list(chains))

        expected_chain = [Chain([self.s3_c_d, self.s3_d_e])]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('E,'), max_len=2)
        self.assertListEqual(expected_chain, list(chains))

    def test_chains_with_negated_asserts(self):
        s1_not_c_b = sym_schema('~C,/A1/B,', reliability=1.0)

        self.tree.add_context_spin_offs(self.s1_b, [s1_not_c_b])

        # test: negated context
        expected_chains = [
            Chain([self.s3_c_d, self.s3_d_e, s1_not_c_b]),
            Chain([self.s3_c_d, s1_not_c_b]),
            Chain([self.s1_a_b]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('B,'))
        self.assertSetEqual(set(expected_chains), set(chains))

        # test: negated context
        expected_chains = [
            Chain([self.s1_a_b, self.s2_b_c]),
            Chain([self.s3_d_e, s1_not_c_b, self.s2_b_c]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('C,'))
        self.assertSetEqual(set(expected_chains), set(chains))

        expected_chains = [
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d]),
            Chain([s1_not_c_b, self.s2_b_c, self.s3_c_d]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('D,'))
        self.assertSetEqual(set(expected_chains), set(chains))

        # test: negated context and result
        s2_not_c = sym_schema('/A2/~C,', reliability=0.25)
        s2_e_not_c = sym_schema('E,/A2/~C,', reliability=1.0)

        self.tree.add_result_spin_offs(self.s2, [s2_not_c])
        self.tree.add_context_spin_offs(s2_not_c, [s2_e_not_c])

        expected_chains = [
            Chain([self.s3_c_d, self.s3_d_e, s2_e_not_c, s1_not_c_b]),
            Chain([self.s3_c_d, self.s3_d_e, s1_not_c_b]),
            Chain([self.s3_c_d, s1_not_c_b]),
            Chain([self.s1_a_b]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('B,'))
        self.assertSetEqual(set(expected_chains), set(chains))

    def test_chains_with_negated_goal_state(self):
        expected_chains = [
            Chain([self.s1_a_b]),
            Chain([self.s1_a_b, self.s2_b_c]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('~A,'))
        self.assertSetEqual(set(expected_chains), set(chains))

        expected_chains = [
            Chain([self.s1_a_b, self.s2_b_c]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('~B,'))
        self.assertSetEqual(set(expected_chains), set(chains))

        expected_chains = [
            Chain([self.s1_a_b]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('~C,'))
        self.assertSetEqual(set(expected_chains), set(chains))

        expected_chains = [
            Chain([self.s1_a_b]),
            Chain([self.s1_a_b, self.s2_b_c]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d, self.s3_d_e]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('~D,'))
        self.assertSetEqual(set(expected_chains), set(chains))

        expected_chains = [
            Chain([self.s1_a_b]),
            Chain([self.s1_a_b, self.s2_b_c]),
            Chain([self.s1_a_b, self.s2_b_c, self.s3_c_d]),
        ]
        chains = self.sm.backward_chains(goal_state=sym_state_assert('~E,'))
        self.assertSetEqual(set(expected_chains), set(chains))
