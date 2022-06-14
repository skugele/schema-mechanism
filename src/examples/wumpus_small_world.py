import argparse
import logging.config
from pathlib import Path
from statistics import mean
from time import sleep
from typing import Iterable
from typing import Optional

from examples import RANDOM_SEED
from examples import display_item_values
from examples import display_known_schemas
from examples import display_summary
from examples import is_paused
from examples import is_running
from examples import run_decorator
from examples.environments.wumpus_world import WumpusWorldAgent
from examples.environments.wumpus_world import WumpusWorldMDP
from schema_mechanism.core import EligibilityTraceDelegatedValueHelper
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import sym_item
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import init
from schema_mechanism.modules import load
from schema_mechanism.modules import save
from schema_mechanism.parameters import GlobalParams
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.trace import ReplacingTrace
from schema_mechanism.util import set_random_seed

logger = logging.getLogger('examples.environments.wumpus_small_world')

# For reproducibility, we we also need to set PYTHONHASHSEED=RANDOM_SEED in the environment
set_random_seed(RANDOM_SEED)

MAX_EPISODES = 50
MAX_STEPS = 25

SERIALIZATION_ENABLED = True
SAVE_DIR: Path = Path('./local/save/wumpus_world')


def init_params() -> GlobalParams:
    params = get_global_params()

    params.set('learning_rate', 0.01)

    params.set('schema.reliability_threshold', 0.9)

    params.set('composite_actions.backward_chains.max_length', 3)
    params.set('composite_actions.update_frequency', 0.005)
    params.set('composite_actions.min_baseline_advantage', 0.5)

    params.set('ext_context.correlation_test', 'FisherExactCorrelationTest')
    params.set('ext_context.positive_correlation_threshold', 0.95)
    params.set('ext_result.correlation_test', 'CorrelationOnEncounter')
    params.set('ext_result.positive_correlation_threshold', 0.8)

    return params


def create_schema_mechanism(env: WumpusWorldMDP) -> SchemaMechanism:
    schema_mechanism: Optional[SchemaMechanism] = None

    if SERIALIZATION_ENABLED:
        try:
            schema_mechanism = load(SAVE_DIR)
        except FileNotFoundError as e:
            logger.warning(e)

    if not schema_mechanism:
        delegated_value_helper = EligibilityTraceDelegatedValueHelper(
            discount_factor=0.5,
            eligibility_trace=ReplacingTrace(
                active_value=1.0,
                decay_strategy=GeometricDecayStrategy(rate=0.1)
            )
        )

        primitive_items = [
            sym_item('EVENT[AGENT ESCAPED]', primitive_value=1.0),
            # sym_item('(EVENT[AGENT ESCAPED],AGENT.HAS[GOLD])', primitive_value=1.0),
        ]

        schema_mechanism = init(
            items=primitive_items,
            actions=env.actions,
            delegated_value_helper=delegated_value_helper,
            global_params=init_params()
        )

    return schema_mechanism


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
    # wumpus_spec = Wumpus(position=(3, 1), health=2)

    world = """
    wwwwww
    w....w
    w.ww.w
    wg..ew
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


@run_decorator
def run() -> None:
    args = parse_args()

    max_steps = args.steps
    max_episodes = args.episodes

    env = create_world()
    schema_mechanism = create_schema_mechanism(env)

    render_env = True

    steps_in_episode = []
    for episode in range(1, max_episodes + 1):

        # initialize the world
        selection_state, is_terminal = env.reset()

        step: int = 0
        for step in range(1, max_steps + 1):
            if is_terminal or not is_running():
                break

            progress_id = f'{episode}:{step}'

            if render_env:
                logger.info(f'\n{env.render()}\n')

            logger.info(f'state [{progress_id}]: {selection_state}')
            selection_details = schema_mechanism.select(selection_state)

            current_composite_schema = schema_mechanism.schema_selection.pending_schema
            if current_composite_schema:
                logger.info(f'active composite action schema [{progress_id}]: {current_composite_schema} ')

            terminated_composite_schemas = selection_details.terminated_pending
            if terminated_composite_schemas:
                logger.info(f'terminated schemas:')
                for i, pending_details in enumerate(terminated_composite_schemas):
                    logger.info(f'schema [{i}]: {pending_details.schema}')
                    logger.info(f'selection state [{i}]: {pending_details.selection_state}')
                    logger.info(f'status [{i}]: {pending_details.status}')
                    logger.info(f'duration [{i}]: {pending_details.duration}')

            schema = selection_details.selected
            action = schema.action
            effective_value = selection_details.effective_value

            logger.info(f'selected schema [{progress_id}]: {schema} [eff. value: {effective_value}]')

            result_state, is_terminal = env.step(action)

            schema_mechanism.learn(selection_details, result_state=result_state)

            if is_paused():
                display_item_values()
                display_known_schemas(schema_mechanism)
                # display_schema_info(sym_schema(f'/USE[EXIT]/'))
                if episode > 1:
                    display_performance_summary(max_steps, steps_in_episode)

                try:
                    while is_paused():
                        sleep(0.1)
                except KeyboardInterrupt:
                    pass

            selection_state = result_state

        steps_in_episode.append(step)
        display_performance_summary(max_steps, steps_in_episode)

        # GlobalStats().delegated_value_helper.eligibility_trace.clear()

    display_summary(schema_mechanism)

    if SERIALIZATION_ENABLED:
        save(
            modules=[schema_mechanism],
            path=SAVE_DIR,
        )


def display_performance_summary(max_steps: int, steps_in_episode: Iterable[int]) -> None:
    steps_in_episode = list(steps_in_episode)

    steps_in_last_episode = steps_in_episode[-1]
    n_episodes = len(steps_in_episode)
    n_episodes_agent_escaped = sum(1 for steps in steps_in_episode if steps < max_steps)
    avg_steps_per_episode = mean(steps_in_episode)
    min_steps_per_episode = min(steps_in_episode)
    max_steps_per_episode = max(steps_in_episode)
    total_steps = sum(steps_in_episode)

    logger.info(f'**** EPISODE {n_episodes} SUMMARY ****')
    logger.info(f'\tepisodes: {n_episodes}')
    logger.info(f'\tepisodes in which agent escaped: {n_episodes_agent_escaped}')
    logger.info(f'\tcumulative steps: {total_steps}')
    logger.info(f'\tsteps in last episode: {steps_in_last_episode}')
    logger.info(f'\taverage steps per episode: {avg_steps_per_episode}')
    logger.info(f'\tminimum steps per episode: {min_steps_per_episode}')
    logger.info(f'\tmaximum steps per episode: {max_steps_per_episode}')


if __name__ == "__main__":
    # TODO: Add code to load a serialized instance that was saved to disk
    # configure logger
    logging.config.fileConfig('config/logging.conf')

    run()
