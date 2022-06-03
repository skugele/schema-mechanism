from typing import Any
from unittest import TestCase

from schema_mechanism.core import Action
from schema_mechanism.core import NULL_CONTROLLER
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.serialization.json.decoders import decode
from schema_mechanism.serialization.json.encoders import encode
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestAction(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.a = Action(label='label')

    def test_init(self):
        self.assertEqual('label', self.a.label)

        # test: a globally unique id (uid) should be assigned for each action
        self.assertIsNotNone(self.a.uid)

        # uses a set to test uniqueness of a large number of actions (set size should equal to the number of actions)
        n_actions = 100_000
        self.assertTrue(n_actions, len(set([Action() for _ in range(n_actions)])))

    def test_non_composite_action_goal_state(self):
        non_composite_action = Action()

        # test: a non-composite action's goal state should be the null state assertion
        self.assertIs(NULL_STATE_ASSERT, non_composite_action.goal_state)

    def test_non_composite_action_controller(self):
        non_composite_action = Action()

        # test: a non-composite action's controller should be the null controller
        self.assertIs(NULL_CONTROLLER, non_composite_action.controller)

    def test_equals(self):
        self.assertTrue(satisfies_equality_checks(obj=self.a, other=Action('other'), other_different_type=1.0))

        # test: labels should be used to determine equality (if they exist)
        self.assertEqual(Action('label 1'), Action('label 1'))
        self.assertNotEqual(Action('label 1'), Action('label 2'))
        self.assertNotEqual(Action(), Action())

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.a))

    def test_encode_and_decode(self):
        # test: encoding/decoding of Actions with labels (should result in Actions with the same labels)
        action = Action(label='test action')

        object_registry: dict[int, Any] = dict()

        json_str = encode(action, object_registry=object_registry)
        recovered = decode(json_str, object_registry=object_registry)

        self.assertIsInstance(recovered, Action)
        self.assertEqual(action.label, recovered.label)

        # test: encoding/decoding of Actions without labels (may result in actions with different uids)
        action = Action()

        object_registry: dict[int, Any] = dict()

        json_str = encode(action, object_registry=object_registry)
        recovered = decode(json_str, object_registry=object_registry)

        self.assertIsInstance(recovered, Action)
