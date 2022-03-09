import unittest

from schema_mechanism.core import CompositeAction
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import SchemaTree
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import SchemaMemory
from test_share.test_classes import MockSchema
from test_share.test_func import common_test_setup


class TestCompositeAction(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # allows direct setting of reliability (only reliable schemas are eligible for chaining)
        GlobalParams().set('schema_type', MockSchema)

        # construct a SchemaTree for testing proximity to goal states
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

        self.tree.add_result_spin_offs(self.s1, [self.s1_b])
        self.tree.add_result_spin_offs(self.s2, [self.s2_c])
        self.tree.add_result_spin_offs(self.s3, [self.s3_d, self.s3_e])

        self.tree.add_context_spin_offs(self.s1_b, [self.s1_a_b])
        self.tree.add_context_spin_offs(self.s2_c, [self.s2_b_c])
        self.tree.add_context_spin_offs(self.s3_d, [self.s3_c_d])
        self.tree.add_context_spin_offs(self.s3_e, [self.s3_d_e])

        self.sm = SchemaMemory.from_tree(self.tree)

        self.goal_state = sym_state_assert('E')
        self.label = 'CA1'

        self.ca = CompositeAction(goal_state=self.goal_state, label=self.label)

    # noinspection PyTypeChecker,PyArgumentList
    def test_init(self):
        # test: the goal state should be required and set properly
        self.assertRaises(TypeError, lambda: CompositeAction())
        self.assertRaises(ValueError, lambda: CompositeAction(goal_state=None))

        # test: an optional label should be set if provided
        self.assertEqual(self.label, self.ca.label)
        self.assertIsNone(CompositeAction(self.goal_state).label)

        # test: a unique id should be assigned
        self.assertIsNotNone(self.ca.uid)

    def test_proximity(self):
        # test: the goal proximity of all schemas should be np.inf
        pass

        # TODO: need to access a list of schemas
        # TODO: need a controller method to check proximity
        # TODO: need a controller method to set proximity

    def test_components(self):
        chains = self.sm.backward_chains(goal_state=self.goal_state)
        print(chains)


class TestController(unittest.TestCase):

    # TODO: This replicates the majority of the setup from TestCompositeAction. Need to move the common
    # TODO: code into a shared function.
    def setUp(self) -> None:
        common_test_setup()

        # allows direct setting of reliability (only reliable schemas are eligible for chaining)
        GlobalParams().set('schema_type', MockSchema)

        # construct a SchemaTree for testing proximity to goal states
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

        self.tree.add_result_spin_offs(self.s1, [self.s1_b])
        self.tree.add_result_spin_offs(self.s2, [self.s2_c])
        self.tree.add_result_spin_offs(self.s3, [self.s3_d, self.s3_e])

        self.tree.add_context_spin_offs(self.s1_b, [self.s1_a_b])
        self.tree.add_context_spin_offs(self.s2_c, [self.s2_b_c])
        self.tree.add_context_spin_offs(self.s3_d, [self.s3_c_d])
        self.tree.add_context_spin_offs(self.s3_e, [self.s3_d_e])

        self.sm = SchemaMemory.from_tree(self.tree)

        self.controller = CompositeAction.Controller()

    def test_init(self):
        # test: components should be an empty set
        self.assertSetEqual(set(), set(self.controller.components))

    def test_update(self):
        # test: goal state with no known chains should result in empty components
        pass

        # test: schemas in backward chains from goal state should be added as components after update

        # TODO: this should actually be based on duration (could set duration to 1 initally)
        # test: component proximity should be the reciprocal of the distance from goal state
