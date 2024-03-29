from unittest import TestCase

import numpy as np

from schema_mechanism.util import AssociativeArrayList
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks


class TestAssociativeArrayList(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.array_list = AssociativeArrayList()

    def test_init(self):
        pre_allocated = 5000
        block_size = 50

        arr_list = AssociativeArrayList(pre_allocated=pre_allocated, block_size=block_size)

        self.assertEqual(pre_allocated, arr_list.n_allocated)
        self.assertEqual(block_size, arr_list.block_size)

    def test_get(self):
        element = '1'

        # sanity check: element should not be in array list
        self.assertNotIn(element, self.array_list)

        # test: get on unknown element should add element with 0.0 value
        len_before_get = len(self.array_list)
        value = self.array_list[element]
        len_after_get = len(self.array_list)

        self.assertIn(element, self.array_list)
        self.assertEqual(0.0, value)
        self.assertEqual(len_before_get + 1, len_after_get)

    def test_set(self):
        element = '1'
        value = -10.7

        # sanity check: element should not be in array list
        self.assertNotIn(element, self.array_list)

        # test: setting value on unknown element should add element with that value
        len_before_set = len(self.array_list)
        self.array_list[element] = value
        len_after_set = len(self.array_list)

        self.assertIn(element, self.array_list)
        self.assertEqual(value, self.array_list[element])
        self.assertEqual(len_before_set + 1, len_after_set)

        # test: setting value on a known element should update its value
        new_value = 7.2

        self.array_list[element] = new_value
        self.assertEqual(new_value, self.array_list[element])

    def test_len(self):
        # test: len of empty array list should be zero
        self.assertEqual(0, len(self.array_list))

        # test: len should increase by one for each added element
        elements = [str(i) for i in range(100)]
        values = [i for i in range(100)]

        expected_len = 0
        for e, v in zip(elements, values):
            self.array_list[e] = v
            expected_len += 1
            self.assertEqual(expected_len, len(self.array_list))

    def test_contains(self):
        arr_list = AssociativeArrayList()

        elements = [str(i) for i in range(10)]
        arr_list.add(elements)

        for e in elements:
            self.assertIn(e, arr_list)

        elements_not_in = [str(i) for i in range(10, 20)]
        for e in elements_not_in:
            self.assertNotIn(e, arr_list)

    def test_equals(self):
        array_list = AssociativeArrayList(pre_allocated=170, block_size=10)
        array_list.add([1, 2, 3, 4, 5])

        other = AssociativeArrayList(pre_allocated=50, block_size=100)
        other.add([2, 3, 4, 5, 6])

        self.assertTrue(satisfies_equality_checks(obj=array_list, other_same_type=other, other_different_type=1.0))

        # test: different pre_allocated values should result in match_strategy returning False
        array_list_1 = AssociativeArrayList(pre_allocated=100, block_size=50)
        array_list_2 = AssociativeArrayList(pre_allocated=90, block_size=50)

        self.assertNotEqual(array_list_1, array_list_2)

        # test: different block_size should result in match_strategy returning False
        array_list_1 = AssociativeArrayList(pre_allocated=100, block_size=50)
        array_list_2 = AssociativeArrayList(pre_allocated=100, block_size=60)

        self.assertNotEqual(array_list_1, array_list_2)

        # test: different indexes should result in match_strategy returning False
        array_list_1 = AssociativeArrayList(pre_allocated=100, block_size=50)
        array_list_1.indexes(['1', '2', '3', '4'], add_missing=True)

        array_list_2 = AssociativeArrayList(pre_allocated=100, block_size=50)
        array_list_2.indexes(['2', '3', '4'], add_missing=True)

        self.assertNotEqual(array_list_1, array_list_2)

        # test: different values should result in match_strategy returning False
        array_list_1 = AssociativeArrayList(pre_allocated=100, block_size=50)
        array_list_1['1'] = 100
        array_list_1['2'] = 100
        array_list_1['3'] = 100

        array_list_2 = AssociativeArrayList(pre_allocated=100, block_size=50)
        array_list_2['1'] = 90
        array_list_2['2'] = 100
        array_list_2['3'] = 75

        self.assertNotEqual(array_list_1, array_list_2)

        # test: these array lists SHOULD be equal
        array_list_1 = AssociativeArrayList(pre_allocated=100, block_size=50)
        array_list_1['1'] = 72
        array_list_1['2'] = -14
        array_list_1['3'] = 25

        array_list_2 = AssociativeArrayList(pre_allocated=100, block_size=50)
        array_list_2['1'] = 72
        array_list_2['2'] = -14
        array_list_2['3'] = 25

        self.assertEqual(array_list_1, array_list_2)

    def test_del(self):
        # test: del for non-existent elements should raise IndexError
        with self.assertRaises(IndexError):
            del self.array_list['non-existent']

        elements = [str(i) for i in range(100)]
        values = [i for i in range(100)]

        for e, v in zip(elements, values):
            self.array_list[e] = v

        # test: del should work for last element in array list (element does not exist after del, len reduced by 1)
        del self.array_list[elements[-1]]

        self.assertNotIn(elements[-1], self.array_list)
        self.assertEqual(99, len(self.array_list))

        # test: del should work for first element in array list (element does not exist after del, len reduced by 1)
        del self.array_list[elements[0]]

        self.assertNotIn(elements[0], self.array_list)
        self.assertEqual(98, len(self.array_list))

        # test: del should work for middle element in array list (element does not exist after del, len reduced by 1)
        del self.array_list[elements[50]]

        self.assertNotIn(elements[50], self.array_list)
        self.assertEqual(97, len(self.array_list))

        # test: removing all elements in array list
        for e in elements:
            if e in self.array_list:
                del self.array_list[e]
                self.assertNotIn(e, self.array_list)

        self.assertTrue(all({e not in self.array_list for e in elements}))
        self.assertEqual(0, len(self.array_list))

    def test_items(self):
        # test: items over empty array list should stop iteration immediately
        for _, _ in self.array_list.items():
            self.fail('Unexpected iteration on empty array list')

        # test: items over non-empty array list should visit all keys and return their corresponding values
        elements = [str(i) for i in range(100)]
        values = [i for i in range(100)]

        for e, v in zip(elements, values):
            self.array_list[e] = v

        encountered_keys = set()
        for e, v in self.array_list.items():
            encountered_keys.add(e)
            self.assertEqual(v, values[elements.index(e)])

        self.assertSetEqual(set(elements), encountered_keys)

    def test_indexes(self):
        # test: indexes on empty list of elements should return empty list
        self.assertListEqual([], list(self.array_list.indexes([])))

        # test: indexes on non-existent elements should return ValueError (add_missing == False)
        self.assertRaises(ValueError, lambda: self.array_list.indexes(['non-existent'], add_missing=False))

        elements = [str(i) for i in range(10)]
        values = [i for i in range(10)]

        for e, v in zip(elements, values):
            self.array_list[e] = v

        # test: indexes on elements that exist should return their respective indexes
        indexes = self.array_list.indexes(elements)
        self.assertTrue(np.array_equal(np.array(values), self.array_list.values[indexes]))

        # test: mixture of existing and non-existing keys should return a ValueError
        self.assertRaises(ValueError, lambda: self.array_list.indexes(['1', '2', 'non-existent', '3']))

    def test_indexes_with_add_missing_1(self):
        unknown_keys = ['unk1', 'unk2', 'unk3']

        try:

            # test: getting index of unknown elements SHOULD NOT return a value error
            indexes = self.array_list.indexes(keys=unknown_keys, add_missing=True)

            # test: returned indexes should be unique
            self.assertEqual(len(unknown_keys), len(set(indexes)))

            # test: elements should now be contained in array list
            for k in unknown_keys:
                self.assertIn(k, self.array_list)

            # test: values should have been expanded for the new elements
            self.assertEqual(len(unknown_keys), len(self.array_list))

            # set values
            self.array_list['unk1'] = 1.0
            self.array_list['unk2'] = 2.0
            self.array_list['unk3'] = 3.0

            # test: returned indexes should correspond to the values of the new elements
            self.assertTrue(np.array_equal(self.array_list.values[indexes], np.array([1.0, 2.0, 3.0])))

            # test: mixture of existing and non-existing keys with add_missing == True should add indexes for missing
            #     : and return correct indexes for pre-existing
            known_keys = [*unknown_keys]
            unknown_keys = ['unk4', 'unk5']

            query_keys = [*known_keys, *unknown_keys]

            mixed_indexes = self.array_list.indexes(keys=query_keys, add_missing=True)

            # test: returned indexes should be unique
            self.assertEqual(len(query_keys), len(set(mixed_indexes)))

            # test: elements should now be contained in array list
            for k in query_keys:
                self.assertIn(k, self.array_list)

            # test: previous keys still have same values
            indexes = self.array_list.indexes(known_keys)
            self.assertTrue(np.array_equal(self.array_list.values[indexes], np.array([1.0, 2.0, 3.0])))

            # test: values should have been expanded for the new elements
            self.assertEqual(len(query_keys), len(self.array_list))

        except ValueError as e:
            self.fail(f'Unexpected ValueError: {str(e)}')

    def test_indexes_with_add_missing_2(self):
        known_keys = ['known1', 'known2']
        self.array_list.add(known_keys)

        unknown_keys = ['unk1', 'unk2', 'unk3']

        try:
            # test: mixture of existing and non-existing keys with add_missing == True should add indexes for missing
            #     : and return correct indexes for pre-existing
            query_keys = [*known_keys, *unknown_keys]
            mixed_indexes = self.array_list.indexes(keys=query_keys, add_missing=True)

            # test: returned indexes should be unique
            self.assertEqual(len(query_keys), len(set(mixed_indexes)))

            # test: all elements should be contained in array list
            for k in query_keys:
                self.assertIn(k, self.array_list)

            # test: values should have been expanded for the new elements
            self.assertEqual(len(query_keys), len(self.array_list))

        except ValueError as e:
            self.fail(f'Unexpected ValueError: {str(e)}')

    # noinspection PyTypeChecker
    def test_indexes_called_with_empty_or_none_keys(self):
        # test: indexes should return np.array([]) if keys argument is None
        np.testing.assert_array_equal(np.array([]), self.array_list.indexes([]))
        np.testing.assert_array_equal(np.array([]), self.array_list.indexes(None))

    def test_update(self):
        # test: update from empty dict should be allowed and change nothing
        self.array_list.update({})
        self.assertEqual(0, len(self.array_list))

        # test: update from empty assoc. array list should be allowed and change nothing
        self.array_list.update(AssociativeArrayList())
        self.assertEqual(0, len(self.array_list))

        # test: update from dict with single pair should add that key/value
        self.array_list.update({'1': 1.0})

        self.assertEqual(1, len(self.array_list))
        self.assertIn('1', self.array_list)
        self.assertEqual(1.0, self.array_list['1'])

        # test: update from dict with single pair should add that key/value
        other = AssociativeArrayList()
        other['2'] = 2.0

        self.array_list.update(other)
        self.assertEqual(2, len(self.array_list))
        self.assertIn('2', self.array_list)
        self.assertEqual(2.0, self.array_list['2'])

        # test: update from multi-element dict
        d = {'3': 3.0, '4': 4.0, '5': 5.0}

        self.array_list.update(d)

        self.assertEqual(5, len(self.array_list))

        for e, v in d.items():
            self.assertIn(e, self.array_list)
            self.assertEqual(v, self.array_list[e])

        other = AssociativeArrayList()

        other['6'] = 6.0
        other['7'] = 7.0
        other['8'] = 8.0

        self.array_list.update(other)

        self.assertEqual(8, len(self.array_list))

        for e, v in other.items():
            self.assertIn(e, self.array_list)
            self.assertEqual(v, self.array_list[e])

    def test_iter(self):
        keys = [str(i) for i in range(10)]
        self.array_list.add(keys)

        keys_iterated_over = [e for e in self.array_list]
        self.assertListEqual(keys, keys_iterated_over)

    def test_keys(self):
        # test: empty array list should return an empty array
        self.assertTrue(np.array_equal(np.array([]), self.array_list.keys()))

        new_keys = np.array([str(i) for i in range(10)])
        new_values = np.array([i for i in range(10)])

        self.array_list.update({k: v for k, v in zip(new_keys, new_values)})

        self.assertTrue(np.array_equal(new_keys, self.array_list.keys()))

    def test_values(self):
        # test: values should return an empty array when used on an empty array list
        self.assertTrue(np.array_equal(np.array([]), self.array_list.values))

        elements = [str(i) for i in range(100)]
        values = [i for i in range(100)]

        for e, v in zip(elements, values):
            self.array_list[e] = v

        # test: values should return all values when used on a non-empty array list
        expected = np.array(values)
        actual = self.array_list.values

        self.assertIsInstance(actual, np.ndarray)
        self.assertTrue(np.array_equal(expected, actual))

        # test: changing elements of returned values array changes corresponding array list
        self.array_list.values *= 0.0

        expected = np.zeros_like(elements, dtype=np.float64)
        actual = self.array_list.values

        self.assertTrue(np.array_equal(expected, actual))

        self.array_list.values[1] = 5.0
        self.assertEqual(5.0, self.array_list.values[1])

    def test_clear(self):
        pre_allocated = 5000
        block_size = 50

        arr_list = AssociativeArrayList(pre_allocated=pre_allocated, block_size=block_size)

        # expand size
        arr_list.add([i for i in range(10_000)])

        # sanity check: array list should have expanded capacity from adds
        self.assertGreater(arr_list.n_allocated, pre_allocated)

        arr_list.clear()

        # test:
        self.assertEqual(pre_allocated, arr_list.n_allocated)

    def test_expand_allocated(self):
        pre_allocated = 1000
        arr_list = AssociativeArrayList(pre_allocated=pre_allocated)

        # sanity check
        self.assertEqual(pre_allocated, arr_list.n_allocated)

        # test: no increase in size until 1001 elements are added
        arr_list.add([i for i in range(pre_allocated)])

        self.assertEqual(pre_allocated, arr_list.n_allocated)

        arr_list.add([pre_allocated + 1])
        self.assertTrue(pre_allocated + arr_list.block_size, arr_list.n_allocated)

    def test_bool(self):

        # sanity check: array list should be empty
        self.assertEqual(0, len(self.array_list))

        # test: array list should return True when empty (different from default behavior, which would return False)
        self.assertTrue(bool(self.array_list))

        # test: array list should return True when non-empty
        self.array_list.add([1, 2, 3])
        self.assertTrue(bool(self.array_list))
