import unittest
from typing import Hashable

import numpy as np

from schema_mechanism.validate import BlackListValidator
from schema_mechanism.validate import MultiValidator
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import TypeValidator
from schema_mechanism.validate import WhiteListValidator
from test_share.test_func import common_test_setup


class TestRangeValidator(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.low = -10.0
        self.high = 10.0
        self.exclude = {self.high}

        # Range validator for range [low, high)
        self.validator = RangeValidator(low=self.low, high=self.high, exclude=[self.high])

    def test_init(self):
        # test: attributes should have been set during initialization
        self.assertEqual(self.low, self.validator.low)
        self.assertEqual(self.high, self.validator.high)
        self.assertSetEqual(self.exclude, self.validator.exclude)

        # test: low or high (or both) can be omitted
        try:
            # test: if low value is omitted, it should be replaced by -np.inf
            validator = RangeValidator(low=None, high=10.0)
            self.assertEqual(-np.inf, validator.low)

            # test: if high value is omitted, it should be replaced by np.inf
            validator = RangeValidator(low=-10.0, high=None)
            self.assertEqual(np.inf, validator.high)

            # test: if both omitted, the range should be (-np.inf, np.inf)
            for v in [RangeValidator(), RangeValidator(low=None, high=None)]:
                self.assertEqual(-np.inf, v.low)
                self.assertEqual(np.inf, v.high)

        except ValueError as e:
            self.fail(f'unexpected ValueError: {str(e)}')

        # test: range's low value must be strictly less than its high value
        self.assertRaises(ValueError, lambda: RangeValidator(low=0.0, high=-1.0))
        self.assertRaises(ValueError, lambda: RangeValidator(low=1.0, high=-1.0))
        self.assertRaises(ValueError, lambda: RangeValidator(low=1.0, high=0.5))

    def test_call(self):
        # test: value in range (and not in exclude list) should be accepted
        try:
            for value in range(int(self.low), int(self.high)):
                self.validator(value)

        except ValueError as e:
            self.fail(f'unexpected ValueError: {str(e)}')

        # test: value less than low range should raise a ValueError
        self.assertRaises(ValueError, lambda: self.validator(self.low - 1.0))

        # test: value greater than high range should raise a ValueError
        self.assertRaises(ValueError, lambda: self.validator(self.high + 1.0))

        # test: value equal to an excluded value should raise a ValueError
        self.assertRaises(ValueError, lambda: self.validator(next(iter(self.exclude))))


class TestTypeValidator(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.accept_set = {int, float}
        self.validator = TypeValidator(self.accept_set)

    def test_init(self):
        # test: attributes should have been set during initialization
        self.assertSetEqual(self.accept_set, self.validator.accept_set)

        # test: if accept_set is empty it should generate a ValueError
        self.assertRaises(ValueError, lambda: TypeValidator(accept_set=[]))

    def test_call(self):

        try:
            # test: objects of types in accept_set should not generate a ValueError
            for value in [1, 1.0]:
                self.validator(value)

            # test: subclasses of accepted types should be accepted
            class IntSub(int):
                def __new__(cls, i):
                    return super().__new__(cls, i)

            self.validator(IntSub(1))

            # test: None should be accepted
            self.validator(None)
        except ValueError as e:
            self.fail(f'Unexpected ValueError: {str(e)}')

        # test: objects of unaccepted types should generate a ValueError
        self.assertRaises(ValueError, lambda: self.validator('1'))
        self.assertRaises(ValueError, lambda: self.validator(set()))


class TestWhiteListValidator(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.accept_set = set(range(10))
        self.validator = WhiteListValidator(self.accept_set)

    def test_init(self):
        # test: attributes should have been set during initialization
        self.assertSetEqual(self.accept_set, self.validator.accept_set)

        # test: if accept_set is empty it should generate a ValueError
        self.assertRaises(ValueError, lambda: WhiteListValidator(accept_set=[]))

    def test_call(self):
        # test: all values in accept_set should be allowed
        try:
            for value in self.accept_set:
                self.validator(value)
        except ValueError as e:
            self.fail(f'Unexpected ValueError: {str(e)}')

        for illegal_value in [-10.0, 10, 'bad']:
            self.assertRaises(ValueError, lambda: self.validator(illegal_value))

        # test: None should not be allowed unless it is in the accept_set
        self.assertRaises(ValueError, lambda: self.validator(None))


class TestBlackListValidator(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.reject_set = {8, 9}
        self.validator = BlackListValidator(self.reject_set)

    def test_init(self):
        # test: attributes should have been set during initialization
        self.assertSetEqual(self.reject_set, self.validator.reject_set)

        # test: if accept_set is empty it should generate a ValueError
        self.assertRaises(ValueError, lambda: BlackListValidator(reject_set=[]))

    def test_call(self):
        # test: all values not in reject_set should be allowed
        try:
            value: Hashable
            for value in [None, 'accept', 7, 10, 0.0, frozenset()]:
                self.validator(value)

        except ValueError as e:
            self.fail(f'Unexpected ValueError: {str(e)}')

        # test: all values in reject_set should raise a ValueError
        for illegal_value in self.validator.reject_set:
            self.assertRaises(ValueError, lambda: self.validator(illegal_value))


class TestMultiValidator(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.validator_set = {
            TypeValidator(accept_set=[int]),
            RangeValidator(low=0.0, high=10.0, exclude=[10.0]),
        }

        self.validator = MultiValidator(self.validator_set)

    def test_init(self):
        # test: attributes should have been set during initialization
        self.assertSetEqual(self.validator_set, self.validator.validators)

        # test: if accept_set is empty it should generate a ValueError
        self.assertRaises(ValueError, lambda: MultiValidator(validators=[]))

    def test_call(self):
        # test: value should be accepted if it passes validation for all validators
        try:
            for value in range(10):
                self.validator(value)
        except ValueError as e:
            self.fail(f'Unexpected ValueError: {str(e)}')

        # test: value should raise ValueError if it fails any component validations
        self.assertRaises(ValueError, lambda: self.validator(-1))
        self.assertRaises(ValueError, lambda: self.validator(10))
        self.assertRaises(ValueError, lambda: self.validator(0.0))
        self.assertRaises(ValueError, lambda: self.validator('bad'))
