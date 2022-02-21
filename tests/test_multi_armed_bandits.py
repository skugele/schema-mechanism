import unittest
from collections import Counter

from schema_mechanism.core import Action
from schema_mechanism.examples.multi_arm_bandits import BanditEnvironment
from schema_mechanism.examples.multi_arm_bandits import Machine
from schema_mechanism.func_api import sym_state
from test_share.test_func import common_test_setup


class TestMultiArmedBandits(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.machine_always_win = Machine('0', p_win=1.0)
        self.machine_always_lose = Machine('1', p_win=0.0)
        self.machine_even_chance = Machine('2', p_win=0.5)

        self.machines = [self.machine_always_win, self.machine_always_lose, self.machine_even_chance]
        self.env = BanditEnvironment(machines=self.machines)

    def test_init(self):
        # test: machines SHOULD be what was supplied to initializer
        self.assertListEqual(self.machines, list(self.env.machines))

        # test: default initial state SHOULD be 'S' (standing)
        self.assertEqual(sym_state('S'), self.env.current_state)

        # test: sending no machines SHOULD generate a ValueError
        self.assertRaises(ValueError, lambda: BanditEnvironment(machines=[]))

        # test: explicitly passing an initial state SHOULD be reflected in the env's current state
        init_state = sym_state('M0')
        env = BanditEnvironment(self.machines, init_state)
        self.assertEqual(init_state, env.current_state)

        # test: invalid initial states SHOULD generate a ValueError
        self.assertRaises(ValueError, lambda: BanditEnvironment(self.machines, init_state=sym_state('INVALID')))

    def test_machines(self):
        # test: p_win = 1.0 SHOULD always have 'W' outcomes
        counter = Counter(self.machine_always_win.play(100))
        self.assertEqual(100, counter['W'])

        # test: p_win = 0.0 SHOULD always have 'L' outcomes
        counter = Counter(self.machine_always_lose.play(100))
        self.assertEqual(100, counter['L'])

        # test: p_win = 0.5 SHOULD have approximately equal 'W' and 'L' outcomes
        counter = Counter(self.machine_even_chance.play(100_000))
        self.assertAlmostEqual(50_000, counter['L'], delta=2500)
        self.assertAlmostEqual(50_000, counter['W'], delta=2500)

    def test_step_1(self):
        for m_id in [m.id for m in self.machines]:
            # test: S + Action(sit(M[m_id])) => M[m_id]
            self.env.step(Action(f'sit(M{m_id})'))
            self.assertEqual(sym_state(f'M{m_id}'), self.env.current_state)

            # test: M[m_id] + Action(stand) => S
            self.env.step(Action('stand'))
            self.assertEqual(sym_state('S'), self.env.current_state)

    def test_step_2(self):
        for m_id in [m.id for m in self.machines]:
            env = BanditEnvironment(machines=self.machines, init_state=sym_state(f'M{m_id}'))

            # test: M{m_id} + Action('deposit') => M{m_id},P
            env.step(Action('deposit'))
            self.assertEqual(sym_state(f'M{m_id},P'), env.current_state)

    def test_step_3(self):
        env = BanditEnvironment(machines=self.machines, init_state=sym_state('M0,P'))

        # test: M0,P + Action('play') => M0,W
        env.step(Action('play'))
        self.assertEqual(sym_state('M0,W'), env.current_state)

        # test: M0,W + Action('stand') => S
        env.step(Action('stand'))
        self.assertEqual(sym_state('S'), env.current_state)

    def test_step_4(self):
        env = BanditEnvironment(machines=self.machines, init_state=sym_state('M1,P'))

        # test: M1,P + Action('play') => M1,L
        env.step(Action('play'))
        self.assertEqual(sym_state('M1,L'), env.current_state)

        # test: M1,W + Action('stand') => S
        env.step(Action('stand'))
        self.assertEqual(sym_state('S'), env.current_state)

    def test_step_5(self):
        # test: all actions other than sit SHOULD result in NO CHANGE in current state
        env = BanditEnvironment(machines=[self.machine_always_win], init_state=sym_state('S'))
        for action in [Action(a_str) for a_str in ['deposit', 'stand', 'play']]:
            self.assertEqual(sym_state('S'), env.step(action))

    def test_step_6(self):
        # test: all actions other than stand and deposit SHOULD result in NO CHANGE in current state
        env = BanditEnvironment(machines=[self.machine_always_win], init_state=sym_state('M0'))
        for action in [Action(a_str) for a_str in ['play', 'sit(M0)']]:
            self.assertEqual(sym_state('M0'), env.step(action))

    def test_step_7(self):
        # test: all actions other than stand and deposit should result in M0 from lose (W) state
        env = BanditEnvironment(machines=self.machines, init_state=sym_state('M0,W'))
        for action in [Action(a_str) for a_str in ['play', *[f'sit(M{m.id})' for m in self.machines]]]:
            self.assertEqual(sym_state('M0'), env.step(action))

    def test_step_8(self):
        # test: all actions other than stand and deposit should result in M0 from win (L) state
        env = BanditEnvironment(machines=self.machines, init_state=sym_state('M0,L'))
        for action in [Action(a_str) for a_str in ['play', *[f'sit(M{m.id})' for m in self.machines]]]:
            self.assertEqual(sym_state('M0'), env.step(action))

    # TODO: test for invalid moves that return to the same state
