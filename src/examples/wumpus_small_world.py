import argparse
from statistics import mean
from time import sleep
from time import time
from typing import Iterable

from pynput import keyboard

from examples import display_item_values
from examples import display_known_schemas
from examples import display_summary
from examples.environments.wumpus_world import Wumpus
from examples.environments.wumpus_world import WumpusWorldAgent
from examples.environments.wumpus_world import WumpusWorldMDP
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.func_api import sym_item
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.share import debug
from schema_mechanism.share import display_params
from schema_mechanism.share import info
from schema_mechanism.share import trace
from schema_mechanism.stats import CorrelationOnEncounter
from schema_mechanism.stats import FisherExactCorrelationTest
# global constants
from schema_mechanism.strategies.evaluation import ExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import GoalPursuitEvaluationStrategy
from schema_mechanism.strategies.match import AbsoluteDiffMatchStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy

MAX_EPISODES = 5000
MAX_STEPS = 500

pause = False
running = True


def on_press(key):
    global pause, running

    if key == keyboard.Key.space:
        pause = not pause
    if key == keyboard.Key.esc:
        print('setting running to False', flush=True)
        running = False


def create_schema_mechanism(env: WumpusWorldMDP) -> SchemaMechanism:
    primitive_items = [
        sym_item('EVENT[AGENT ESCAPED]', primitive_value=100.0),
    ]
    bare_schemas = [SchemaPool().get(SchemaUniqueKey(action=a)) for a in env.actions]
    schema_memory = SchemaMemory(bare_schemas)
    schema_selection = SchemaSelection(
        select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0)),
        value_strategies=[
            GoalPursuitEvaluationStrategy(),
            ExploratoryEvaluationStrategy(),
        ],
        weights=[0.5, 0.5]
    )

    sm: SchemaMechanism = SchemaMechanism(
        items=primitive_items,
        schema_memory=schema_memory,
        schema_selection=schema_selection)

    sm.params.set('backward_chains.max_len', 5)
    sm.params.set('backward_chains.update_frequency', 0.01)
    sm.params.set('composite_actions.learn.min_baseline_advantage', 10.0)
    sm.params.set('delegated_value_helper.decay_rate', 0.3)
    sm.params.set('delegated_value_helper.discount_factor', 0.8)
    sm.params.set('ext_context.correlation_test', FisherExactCorrelationTest)
    sm.params.set('ext_context.positive_correlation_threshold', 0.95)
    sm.params.set('ext_result.correlation_test', CorrelationOnEncounter)
    sm.params.set('ext_result.positive_correlation_threshold', 0.95)
    sm.params.set('goal_pursuit_strategy.reliability.max_penalty', 5.0)
    sm.params.set('habituation_exploratory_strategy.decay.rate', 0.01)
    sm.params.set('habituation_exploratory_strategy.multiplier', 1.0)
    sm.params.set('learning_rate', 0.01)
    sm.params.set('random_exploratory_strategy.epsilon.decay.rate.initial', 0.99)
    sm.params.set('random_exploratory_strategy.epsilon.decay.rate.min', 0.1)
    sm.params.set('reliability_threshold', 0.9)

    # sm.params.get('features').remove(SupportedFeature.COMPOSITE_ACTIONS)

    return sm


def parse_args():
    """ Parses command line arguments.
    :return: argparse parser with parsed command line args
    """
    parser = argparse.ArgumentParser(description='Wumpus World Example Environment (Schema Mechanism)')

    parser.add_argument('--steps', type=int, required=False, default=MAX_STEPS,
                        help=f'the maximum number of steps before terminating an episode (default: {MAX_STEPS})')
    parser.add_argument('--episodes', type=int, required=False, default=MAX_EPISODES,
                        help=f'the maximum number of episodes (default: {MAX_EPISODES})')

    return parser.parse_args()


def create_world() -> WumpusWorldMDP:
    agent_spec = WumpusWorldAgent(position=(1, 1), direction='N', n_arrows=0)
    wumpus_spec = Wumpus(position=(3, 1), health=2)

    world = """
    wwwwww
    w....w
    w.ww.w
    w...ew
    wwwwww
    """
    # world = """
    # wwwwwwwwww
    # w........w
    # w.wwa.wwww
    # w.ppwwwppw
    # w.pwww..ew
    # w.pgpw.ppw
    # w...w..www
    # w..ww.wwpw
    # w.......aw
    # wwwwwwwwww
    # """

    # note: randomizing the start seems to be extremely important in episodic environments because the SchemaMechanism
    # will learn unintended correlations due to repeated exposure to the same state state. For example, that turning
    # left results in a northerly orientation, or that moving forward (without constraint) leads to a particular
    # position.
    env = WumpusWorldMDP(worldmap=world, agent=agent_spec, wumpus=None, randomized_start=True)
    return env


def run() -> None:
    args = parse_args()

    env = create_world()
    sm = create_schema_mechanism(env)

    display_params()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    start_time = time()

    steps_in_episode = []

    for episode in range(args.episodes):

        # initialize the world
        selection_state, is_terminal = env.reset()

        step: int = 0
        for step in range(args.steps):
            if is_terminal or not running:
                break

            progress_id = f'{episode}:{step}'

            trace(f'\n{env.render()}\n')

            debug(f'state [{progress_id}]: {selection_state}')
            selection_details = sm.select(selection_state)

            current_composite_schema = sm.schema_selection.pending_schema
            if current_composite_schema:
                debug(f'active composite action schema [{progress_id}]: {current_composite_schema} ')

            terminated_composite_schemas = selection_details.terminated_pending
            if terminated_composite_schemas:
                info(f'terminated schemas:')
                for i, pending_details in enumerate(terminated_composite_schemas):
                    debug(f'schema [{i}]: {pending_details.schema}')
                    debug(f'selection state [{i}]: {pending_details.selection_state}')
                    debug(f'status [{i}]: {pending_details.status}')
                    debug(f'duration [{i}]: {pending_details.duration}')

            schema = selection_details.selected
            action = schema.action

            debug(f'selected schema [{progress_id}]: {schema} [eff. value: {selection_details.effective_value}]')

            result_state, is_terminal = env.step(action)

            sm.learn(selection_details, result_state=result_state)

            if pause:
                display_item_values()
                display_known_schemas(sm, composite_only=True)

                try:
                    while pause:
                        sleep(0.1)
                except KeyboardInterrupt:
                    pass

            selection_state = result_state

        steps_in_episode.append(step)
        display_performance_summary(args, steps_in_episode)

        # GlobalStats().delegated_value_helper.eligibility_trace.clear()

    display_run_summary(sm, start_time)

    # TODO: Add code to save a serialized instance of this schema mechanism to disk


def display_run_summary(sm: SchemaMechanism, start_time: float):
    display_summary(sm)

    info(f'terminating execution after {time() - start_time} seconds')


def display_performance_summary(args, steps_in_episode: Iterable[int]) -> None:
    steps_in_episode = list(steps_in_episode)

    steps_in_last_episode = steps_in_episode[-1]
    n_episodes = len(steps_in_episode)
    n_episodes_agent_escaped = sum(1 for steps in steps_in_episode if steps < args.steps)
    avg_steps_per_episode = mean(steps_in_episode)
    min_steps_per_episode = min(steps_in_episode)

    info('**** EPISODE SUMMARY ****')
    info(f'\t# episodes: {n_episodes}')
    info(f'\t# steps in last episode: {steps_in_last_episode}')
    info(f'\t# episodes in which agent escaped: {n_episodes_agent_escaped}')
    info(f'\taverage steps per episode: {avg_steps_per_episode}')
    info(f'\tmin steps per episode: {min_steps_per_episode}')


if __name__ == "__main__":
    # TODO: Add code to load a serialized instance that was saved to disk

    run()
