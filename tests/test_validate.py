from enum import Enum
from enum import auto
from typing import Hashable
from unittest import TestCase

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import Item
from schema_mechanism.core import Schema
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.share import SupportedFeature
from schema_mechanism.validate import AcceptAllValidator
from schema_mechanism.validate import BlackListValidator
from schema_mechanism.validate import ElementWiseValidator
from schema_mechanism.validate import MultiValidator
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import SubClassValidator
from schema_mechanism.validate import SupportedFeatureValidator
from schema_mechanism.validate import TypeValidator
from schema_mechanism.validate import WhiteListValidator
from test_share.test_classes import MockSchema
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestAcceptAllValidator(TestCase):
    def setUp(self):
        common_test_setup()

        self.validator = AcceptAllValidator()

    def test_call(self):
        # test: should NEVER generate a ValueError (accepts everything)
        try:
            self.validator(value=0.0)
            self.validator(value=-1.0)
            self.validator(value=100.0)
            self.validator(value='100')
            self.validator(value=set())
            self.validator(value=[1, 2, 3, 4])
            self.validator(value=sym_schema('/A1/'))
        except ValueError as e:
            self.fail(f'Raised unexpected ValueError: {e}')

    def test_equals(self):
        obj = AcceptAllValidator()
        other = AcceptAllValidator()

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))


class TestRangeValidator(TestCase):
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

        # test: ValueError should be raised if value is non-numeric
        self.assertRaises(ValueError, lambda: self.validator('27'))

    def test_equals(self):
        obj = RangeValidator(low=0.1, high=1.0, exclude=[self.high])
        other = RangeValidator(low=2.0, high=5.0, exclude=[self.low])

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))


class TestTypeValidator(TestCase):
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

    def test_equals(self):
        obj = TypeValidator(accept_set={float})
        other = TypeValidator(accept_set={str})

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))


class TestSubClassValidator(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.accept_set = {Item, Schema}
        self.validator = SubClassValidator(accept_set=self.accept_set)

    # noinspection PyTypeChecker
    def test_init(self):
        # test: attributes should have been set during initialization
        self.assertSetEqual(self.accept_set, self.validator.accept_set)

        # test: if accept_set is empty it should generate a ValueError
        self.assertRaises(ValueError, lambda: SubClassValidator(accept_set=[]))
        self.assertRaises(ValueError, lambda: SubClassValidator(accept_set=None))

    def test_call(self):

        accepted_values = [
            sym_item('1'),  # SymbolicItem
            sym_item('1', item_type=MockSymbolicItem),  # MockSymbolicItem
            sym_item('(1,2)'),  # CompositeItem
            sym_schema('1,/A1/2,'),  # Schema
            sym_schema('1,/A2/3,4', type=MockSchema),  # MockSchema
        ]

        try:

            # test: objects of types or subtypes in accept_set should not generate a ValueError
            for value in accepted_values:
                self.validator(type(value))

            # test: None should be accepted
            self.validator(None)

        except ValueError as e:
            self.fail(f'Unexpected ValueError: {str(e)}')

        rejected_values = [
            1,
            1.2,
            '123',
            [1, 2, 3],
            {1, 2, 3},
            Action()
        ]

        for value in rejected_values:
            # test: objects of unaccepted types should generate a ValueError
            self.assertRaises(ValueError, lambda: self.validator(type(value)))

    def test_equals(self):
        obj = SubClassValidator(accept_set={float})
        other = SubClassValidator(accept_set={str})

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))


class TestWhiteListValidator(TestCase):
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

        # test: non-hashable values should raise a TypeError
        non_hashable_values = [
            list(),
            dict()
        ]
        for value in non_hashable_values:
            self.assertRaises(TypeError, lambda: self.validator(value))

    def test_equals(self):
        obj = WhiteListValidator(accept_set={1, 2, 3})
        other = WhiteListValidator(accept_set={3, 4, 5})

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))


class TestBlackListValidator(TestCase):
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

        # test: non-hashable values should raise a TypeError
        non_hashable_values = [
            list(),
            dict()
        ]
        for value in non_hashable_values:
            self.assertRaises(TypeError, lambda: self.validator(value))

    def test_equals(self):
        obj = BlackListValidator(reject_set={1, 2, 3})
        other = BlackListValidator(reject_set={3, 4, 5})

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))


class TestMultiValidator(TestCase):
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

    def test_equals(self):
        obj = MultiValidator(validators=[RangeValidator(1, 2)])
        other = MultiValidator(validators=[RangeValidator(3, 4)])

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))


class TestElementWiseValidator(TestCase):
    def setUp(self):
        common_test_setup()

        self.validator_arg = RangeValidator(low=0.0, high=1.0)
        self.validator = ElementWiseValidator(validator=self.validator_arg)

    # noinspection PyTypeChecker
    def test_init(self):
        # test: attributes should have been set during initialization
        self.assertEqual(self.validator_arg, self.validator.validator)

        # test: ValueError should be raised if validator parameter is None
        self.assertRaises(ValueError, lambda: ElementWiseValidator(validator=None))

    # noinspection PyTypeChecker
    def test_call(self):

        # test: values that are not iterable should raise a ValueError
        non_iterables = [
            1,
            1.7,
            None
        ]

        for value in non_iterables:
            self.assertRaises(ValueError, lambda: self.validator(value))

        # test: any values accepted by the underlying validator should be accepted by the element-wise validator
        try:
            self.validator(value=np.linspace(self.validator_arg.low, self.validator_arg.high))
        except ValueError as e:
            self.fail(f'Unexpected ValueError raised: {e}')

        # test: any values rejected by the underlying validator should be rejected by the element-wise validator
        self.assertRaises(ValueError, lambda: self.validator(value=np.linspace(-100.0, self.validator_arg.low)))
        self.assertRaises(ValueError, lambda: self.validator(value=np.linspace(self.validator_arg.high, 100.0)))

    def test_equals(self):
        obj = ElementWiseValidator(validator=RangeValidator(1, 2))
        other = ElementWiseValidator(validator=RangeValidator(3, 4))

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))


class TestSupportedFeatureValidator(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.validator = SupportedFeatureValidator()

    # noinspection PyTypeChecker
    def test_call(self):
        supported_features = {feature for feature in SupportedFeature}

        try:
            # test: all features in SupportedFeature enum should pass validator (not raise ValueError)
            self.validator(supported_features)
        except ValueError as e:
            self.fail(f'Unexpected ValueError raised: {e}')

        class NonSupportedFeature(Enum):
            NOT_SUPPORTED_FEATURE_1 = auto()
            NOT_SUPPORTED_FEATURE_2 = auto()
            NOT_SUPPORTED_FEATURE_3 = auto()
            NOT_SUPPORTED_FEATURE_4 = auto()

        # test: all other values (not in SupportedFeature) should be rejected
        non_supported_features = [feature for feature in NonSupportedFeature]
        self.assertRaises(ValueError, lambda: self.validator(non_supported_features))

        # test: ValueError should be raised if EC_MOST_SPECIFIC_ON_MULTIPLE but not EC_DEFER_TO_MORE_SPECIFIC_SCHEMA
        self.assertRaises(ValueError, lambda: self.validator([SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE]))

        all_but_ec_defer_feature = supported_features - {SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA}
        self.assertRaises(ValueError, lambda: self.validator(all_but_ec_defer_feature))

    def test_equals(self):
        obj = SupportedFeatureValidator()
        other = SupportedFeatureValidator()

        self.assertTrue(satisfies_equality_checks(obj=obj, other=other, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.validator))
