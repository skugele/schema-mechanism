import argparse
from statistics import mean
from time import sleep
from typing import Iterable

from examples import display_item_values
from examples import display_known_schemas
from examples import display_summary
from examples import is_paused
from examples import is_running
from examples import run_decorator
from examples.environments.wumpus_world import WumpusWorldAgent
from examples.environments.wumpus_world import WumpusWorldMDP
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.func_api import sym_item
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.share import display_params
from schema_mechanism.share import info
from schema_mechanism.stats import CorrelationOnEncounter
from schema_mechanism.stats import FisherExactCorrelationTest
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.evaluation import CompositeEvaluationStrategy
from schema_mechanism.strategies.evaluation import DelegatedValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import EpsilonGreedyEvaluationStrategy
from schema_mechanism.strategies.evaluation import ReliabilityEvaluationStrategy
from schema_mechanism.strategies.match import AbsoluteDiffMatchStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy

MAX_EPISODES = 5000
MAX_STEPS = 500


def create_schema_mechanism(env: WumpusWorldMDP) -> SchemaMechanism:
    primitive_items = [
        sym_item('EVENT[AGENT ESCAPED]', primitive_value=1.0),
    ]
    bare_schemas = [SchemaPool().get(SchemaUniqueKey(action=a)) for a in env.actions]
    schema_memory = SchemaMemory(bare_schemas)
    schema_selection = SchemaSelection(
        select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.0)),
        evaluation_strategy=CompositeEvaluationStrategy(
            strategies=[
                DelegatedValueEvaluationStrategy(),
                ReliabilityEvaluationStrategy(max_penalty=1e-3),
                EpsilonGreedyEvaluationStrategy(epsilon=0.9999,
                                                epsilon_min=0.05,
                                                decay_strategy=GeometricDecayStrategy(rate=0.9999))
            ]
        )
    )

    sm: SchemaMechanism = SchemaMechanism(
        items=primitive_items,
        schema_memory=schema_memory,
        schema_selection=schema_selection)

    sm.params.set('backward_chains.max_len', 5)
    sm.params.set('backward_chains.update_frequency', 0.01)
    sm.params.set('composite_actions.learn.min_baseline_advantage', 0.1)
    sm.params.set('delegated_value_helper.decay_rate', 0.1)
    sm.params.set('delegated_value_helper.discount_factor', 0.5)
    sm.params.set('ext_context.correlation_test', FisherExactCorrelationTest)
    sm.params.set('ext_context.positive_correlation_threshold', 0.95)
    sm.params.set('ext_result.correlation_test', CorrelationOnEncounter)
    sm.params.set('ext_result.positive_correlation_threshold', 0.95)
    sm.params.set('learning_rate', 0.01)
    sm.params.set('reliability_threshold', 0.9)

    # sm.params.get('features').remove(SupportedFeature.COMPOSITE_ACTIONS)

    display_params()

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
    # wumpus_spec = Wumpus(position=(3, 1), health=2)

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


@run_decorator
def run() -> None:
    args = parse_args()

    max_steps = args.steps
    max_episodes = args.episodes

    env = create_world()
    sm = create_schema_mechanism(env)

    steps_in_episode = []
    for episode in range(1, max_episodes + 1):

        # initialize the world
        selection_state, is_terminal = env.reset()

        step: int = 0
        for step in range(1, max_steps + 1):
            if is_terminal or not is_running():
                break

            progress_id = f'{episode}:{step}'

            info(f'\n{env.render()}\n')

            info(f'state [{progress_id}]: {selection_state}')
            selection_details = sm.select(selection_state)

            current_composite_schema = sm.schema_selection.pending_schema
            if current_composite_schema:
                info(f'active composite action schema [{progress_id}]: {current_composite_schema} ')

            terminated_composite_schemas = selection_details.terminated_pending
            if terminated_composite_schemas:
                info(f'terminated schemas:')
                for i, pending_details in enumerate(terminated_composite_schemas):
                    info(f'schema [{i}]: {pending_details.schema}')
                    info(f'selection state [{i}]: {pending_details.selection_state}')
                    info(f'status [{i}]: {pending_details.status}')
                    info(f'duration [{i}]: {pending_details.duration}')

            schema = selection_details.selected
            action = schema.action
            effective_value = selection_details.effective_value

            info(f'selected schema [{progress_id}]: {schema} [eff. value: {effective_value}]')

            result_state, is_terminal = env.step(action)

            sm.learn(selection_details, result_state=result_state)

            if is_paused():
                display_item_values()
                display_known_schemas(sm)

                try:
                    while is_paused():
                        sleep(0.1)
                except KeyboardInterrupt:
                    pass

            selection_state = result_state

        steps_in_episode.append(step)
        display_performance_summary(max_steps, steps_in_episode)

        # GlobalStats().delegated_value_helper.eligibility_trace.clear()

    display_summary(sm)

    # TODO: Add code to save a serialized instance of this schema mechanism to disk


def display_performance_summary(max_steps: int, steps_in_episode: Iterable[int]) -> None:
    steps_in_episode = list(steps_in_episode)

    steps_in_last_episode = steps_in_episode[-1]
    n_episodes = len(steps_in_episode)
    n_episodes_agent_escaped = sum(1 for steps in steps_in_episode if steps < max_steps)
    avg_steps_per_episode = mean(steps_in_episode)
    min_steps_per_episode = min(steps_in_episode)
    max_steps_per_episode = max(steps_in_episode)
    total_steps = sum(steps_in_episode)

    info(f'**** EPISODE {n_episodes} SUMMARY ****')
    info(f'\tepisodes: {n_episodes}')
    info(f'\tepisodes in which agent escaped: {n_episodes_agent_escaped}')
    info(f'\tcumulative steps: {total_steps}')
    info(f'\tsteps in last episode: {steps_in_last_episode}')
    info(f'\taverage steps per episode: {avg_steps_per_episode}')
    info(f'\tminimum steps per episode: {min_steps_per_episode}')
    info(f'\tmaximum steps per episode: {max_steps_per_episode}')


if __name__ == "__main__":
    # TODO: Add code to load a serialized instance that was saved to disk

    run()
