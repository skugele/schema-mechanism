from unittest import TestCase

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.strategies.decay import ExponentialDecayStrategy
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.trace import AccumulatingTrace
from schema_mechanism.strategies.trace import ReplacingTrace
from schema_mechanism.strategies.trace import Trace
from test_share.test_func import common_test_setup


class TestCommon(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.elements = [Action() for _ in range(100)]
        self.decay_strategy = GeometricDecayStrategy(rate=0.75)

    def assert_trace_length_updates_correctly(self, trace: Trace):
        # test: initial length should be 0
        self.assertEqual(0, len(trace))

        # test: updating with empty set should not increment length of trace
        elements = []
        trace.update(elements)

        self.assertEqual(0, len(trace))

        # test: updating with single new element should increment length by 1
        previous_trace_length = len(trace)
        elements = ['1']
        trace.update(elements)

        self.assertEqual(previous_trace_length + len(elements), len(trace))

        # test: updating with single known element should not increment the length
        previous_trace_length = len(trace)
        elements = ['1']
        trace.update(elements)

        self.assertEqual(previous_trace_length, len(trace))

        # test: updating with another single known element should increment length by 1
        previous_trace_length = len(trace)
        elements = ['2']
        trace.update(elements)

        self.assertEqual(previous_trace_length + len(elements), len(trace))

        # test: updating with multiple new elements should increment length by number of new elements
        previous_trace_length = len(trace)
        elements = ['3', '4', '5']
        trace.update(elements)

        self.assertEqual(previous_trace_length + len(elements), len(trace))

        # test: updating with some new and other known elements should increment length by number of new elements
        previous_trace_length = len(trace)
        new_elements = ['6', '7']
        elements = ['4', '5', *new_elements]
        trace.update(elements)

        self.assertEqual(previous_trace_length + len(new_elements), len(trace))

    def assert_update_increase_values_of_active_set(self, trace: Trace):
        trace.add(self.elements)

        old_values = np.copy(trace.values)
        trace.update(active_set=self.elements)
        new_values = np.copy(trace.values)

        np.testing.assert_array_less(old_values, new_values)

    def assert_values_decay_to_zero(self, trace: Trace):
        trace.add(self.elements)
        trace.update(active_set=self.elements)

        expected_values = np.zeros_like(self.elements, dtype=np.float64)
        for i in range(100_000):
            trace.update(active_set=[])

            # break from loop and return if trace values converge to expected value of all zeros
            if np.allclose(expected_values, trace.values):
                return

        self.fail('Trace values did not decay to zero as expected')

    def assert_allocated_blocks_expands_as_expected(self, trace: Trace):
        n_allocated = trace.n_allocated
        block_size = trace.block_size

        trace.update([i for i in range(block_size + 1)])

        # test: after update, the number of allocated elements should be increased by one
        self.assertTrue(n_allocated + 1, trace.n_allocated)

    def assert_contains_functions_correctly(self, trace: Trace):
        elements = [str(i) for i in range(10)]
        trace.update(elements)

        for e in elements:
            self.assertIn(e, trace)

        elements_not_in = [str(i) for i in range(10, 20)]
        for e in elements_not_in:
            self.assertNotIn(e, trace)

    def assert_clear_functions_correctly(self, trace: Trace):
        trace.update(active_set=self.elements)

        # sanity check: length should be updated to length of elements in active set
        self.assertEqual(len(self.elements), len(trace))

        # sanity check: new elements should be contained in trace
        for element in self.elements:
            self.assertIn(element, trace)

        # sanity check: elements should have non-zero values
        np.testing.assert_array_less(np.zeros_like(self.elements), trace.values)

        trace.clear()

        # test: clear should have removed all elements and length should be zero
        self.assertEqual(0, len(trace))
        for element in self.elements:
            self.assertNotIn(element, trace)

        # test: values array should be an empty array
        np.testing.assert_array_equal(np.array([]), trace.values)

        # adding same elements again to verify that no residual values remain
        trace.add(self.elements)

        # test: adding elements to cleared trace should update length correctly
        self.assertEqual(len(self.elements), len(trace))

        # test: adding elements to cleared trace should result in elements being contained in trace
        for element in self.elements:
            self.assertIn(element, trace)

        # test: adding elements (without update) should result in elements having zero value
        np.testing.assert_array_equal(np.zeros_like(self.elements), trace.values)


class TestAccumulatingTrace(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.active_increment = 0.25

        self.trace = AccumulatingTrace(
            decay_strategy=self.decay_strategy,
            active_increment=self.active_increment
        )

    def test_init(self):
        # parameters for array list
        pre_allocated = 1000
        block_size = 25

        # parameters for trace
        decay_strategy = ExponentialDecayStrategy(rate=0.2, minimum=0.0)
        active_increment = 0.234

        trace = AccumulatingTrace(
            decay_strategy=decay_strategy,
            active_increment=active_increment,
            pre_allocated=pre_allocated,
            block_size=block_size
        )

        self.assertEqual(pre_allocated, trace.n_allocated)
        self.assertEqual(block_size, trace.block_size)
        self.assertEqual(decay_strategy, trace.decay_strategy)
        self.assertEqual(active_increment, trace.active_increment)

    def test_allocated_blocks_expands_as_expected(self):
        self.assert_allocated_blocks_expands_as_expected(self.trace)

    def test_trace_length_updates_correctly(self):
        self.assert_trace_length_updates_correctly(self.trace)

    def test_update_increase_values_of_active_set(self):
        self.assert_update_increase_values_of_active_set(self.trace)

    def test_values_decay_to_zero(self):
        self.assert_values_decay_to_zero(self.trace)

    def test_contains(self):
        self.assert_contains_functions_correctly(self.trace)

    def test_clear(self):
        self.assert_clear_functions_correctly(self.trace)

    def test_update(self):

        # add all elements to trace without update
        self.trace.add(self.elements)

        # sanity check: all elements should now be contained in trace and their values should be all zero
        for element in self.elements:
            self.assertIn(element, self.trace)

        np.testing.assert_array_equal(np.zeros_like(self.elements, dtype=np.float64), self.trace.values)

        active_set = self.elements[:len(self.elements) // 2]
        inactive_set = set(self.elements).difference(active_set)

        active_set_indexes = [self.elements.index(element) for element in active_set]
        inactive_set_indexes = [self.elements.index(element) for element in inactive_set]

        self.trace.update(active_set=active_set)

        # test: initial update should set values in active set to the active_increment value
        expected = self.trace.active_increment * np.ones(len(active_set), dtype=np.float64)
        actual = self.trace.values[active_set_indexes]

        self.assertTrue(np.array_equal(expected, actual))

        # test: elements in inactive set should be zero
        expected = np.zeros(len(inactive_set), dtype=np.float64)
        actual = self.trace.values[inactive_set_indexes]

        self.assertTrue(np.array_equal(expected, actual))

        old_values = np.copy(self.trace.values)
        for _ in range(10):
            self.trace.update(active_set=active_set)
            new_values = np.copy(self.trace.values)

            # test: subsequent updates SHOULD increase active set values
            np.testing.assert_array_less(old_values[active_set_indexes], new_values[active_set_indexes])

            # test: subsequent updates SHOULD NOT increase inactive set values
            np.testing.assert_array_equal(old_values[inactive_set_indexes], new_values[inactive_set_indexes])

            old_values = new_values

    def test_update_converges_to_limit(self):
        # add all elements to trace without update
        self.trace.add(self.elements)

        # sanity check: all elements should now be contained in trace and their values should be all zero
        for element in self.elements:
            self.assertIn(element, self.trace)

        np.testing.assert_array_equal(np.zeros_like(self.elements, dtype=np.float64), self.trace.values)

        values = np.copy(self.trace.values)
        differences = np.inf * np.ones_like(self.elements, dtype=np.float64)

        for _ in range(100):
            self.trace.update(active_set=self.elements)

            new_values = np.copy(self.trace.values)
            new_differences = new_values - values

            # break early if differences close to zero
            if np.allclose(np.zeros_like(self.elements, dtype=np.float64), differences, atol=1e-9):
                break

            np.testing.assert_array_less(new_differences, differences)

            values = new_values
            differences = new_differences

        np.testing.assert_allclose(np.zeros_like(self.elements, dtype=np.float64), differences, atol=1e-9)


class TestReplacingTrace(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.active_value = 0.99

        self.trace = ReplacingTrace(
            decay_strategy=self.decay_strategy,
            active_value=self.active_value
        )

    def test_init(self):
        # parameters for array list
        pre_allocated = 1000
        block_size = 25

        # parameters for trace
        decay_strategy = ExponentialDecayStrategy(rate=0.2, minimum=0.0)
        active_value = 0.234

        trace = ReplacingTrace(
            decay_strategy=decay_strategy,
            active_value=active_value,
            pre_allocated=pre_allocated,
            block_size=block_size
        )

        self.assertEqual(pre_allocated, trace.n_allocated)
        self.assertEqual(block_size, trace.block_size)
        self.assertEqual(decay_strategy, trace.decay_strategy)
        self.assertEqual(active_value, trace.active_value)

    def test_allocated_blocks_expands_as_expected(self):
        self.assert_allocated_blocks_expands_as_expected(self.trace)

    def test_trace_length_updates_correctly(self):
        self.assert_trace_length_updates_correctly(self.trace)

    def test_update_increase_values_of_active_set(self):
        self.assert_update_increase_values_of_active_set(self.trace)

    def test_values_decay_to_zero(self):
        self.assert_values_decay_to_zero(self.trace)

    def test_contains(self):
        self.assert_contains_functions_correctly(self.trace)

    def test_clear(self):
        self.assert_clear_functions_correctly(self.trace)

    def test_update(self):

        # add all elements to trace without update
        self.trace.add(self.elements)

        # sanity check: all elements should now be contained in trace and their values should be all zero
        for element in self.elements:
            self.assertIn(element, self.trace)

        np.testing.assert_array_equal(np.zeros_like(self.elements, dtype=np.float64), self.trace.values)

        active_set = self.elements[:len(self.elements) // 2]
        inactive_set = set(self.elements).difference(active_set)

        active_set_indexes = [self.elements.index(element) for element in active_set]
        inactive_set_indexes = [self.elements.index(element) for element in inactive_set]

        self.trace.update(active_set=active_set)

        # test: initial update should set values in active set to the active_increment value
        expected = self.trace.active_value * np.ones(len(active_set), dtype=np.float64)
        actual = self.trace.values[active_set_indexes]

        self.assertTrue(np.array_equal(expected, actual))

        # test: elements in inactive set should be zero
        expected = np.zeros(len(inactive_set), dtype=np.float64)
        actual = self.trace.values[inactive_set_indexes]

        self.assertTrue(np.array_equal(expected, actual))

        old_values = np.copy(self.trace.values)
        for _ in range(10):
            self.trace.update(active_set=active_set)
            new_values = np.copy(self.trace.values)

            # test: subsequent updates should keep elements fixed (active set == active value; inactive set == 0.0)
            np.testing.assert_array_equal(old_values, new_values)

            old_values = new_values
