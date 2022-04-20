import argparse
from time import sleep
from time import time

from pynput import keyboard

from examples import display_item_values
from examples import display_summary
from examples.environments.wumpus_world import Wumpus
from examples.environments.wumpus_world import WumpusWorldAgent
from examples.environments.wumpus_world import WumpusWorldMDP
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ItemPool
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.func_api import sym_item
from schema_mechanism.modules import AbsoluteDiffMatchStrategy
from schema_mechanism.modules import ExploratoryEvaluationStrategy
from schema_mechanism.modules import GoalPursuitEvaluationStrategy
from schema_mechanism.modules import RandomizeBestSelectionStrategy
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.share import SupportedFeature
from schema_mechanism.share import display_params
from schema_mechanism.share import info
from schema_mechanism.stats import CorrelationOnEncounter
from schema_mechanism.stats import DrescherCorrelationTest

# global constants
N_STEPS = 500
N_EPISODES = 100

# Wumpus @ (6,3); Agent starts @ (1,8) facing 'W'
agent_spec = WumpusWorldAgent(position=(1, 1), direction='N', n_arrows=0)
wumpus_spec = Wumpus(position=(3, 1), health=2)

# world = """
# www
# w.w
# www
# """
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

# global constants
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
        weights=[0.1, 0.9]
    )

    sm: SchemaMechanism = SchemaMechanism(
        items=primitive_items,
        schema_memory=schema_memory,
        schema_selection=schema_selection)

    sm.params.set('backward_chains.max_len', 10)
    sm.params.set('backward_chains.update_frequency', 0.1)
    sm.params.set('composite_action_min_baseline_advantage', 10.0)
    sm.params.set('delegated_value_helper.decay_rate', 0.8)
    sm.params.set('delegated_value_helper.discount_factor', 0.8)
    sm.params.set('ext_context.correlation_test', DrescherCorrelationTest)
    sm.params.set('ext_context.positive_correlation_threshold', 0.95)
    sm.params.set('ext_result.correlation_test', CorrelationOnEncounter)
    sm.params.set('ext_result.positive_correlation_threshold', 0.95)
    sm.params.set('goal_pursuit_strategy.reliability.max_penalty', 5.0)
    sm.params.set('habituation_exploratory_strategy.decay.rate', 0.6)
    sm.params.set('habituation_exploratory_strategy.multiplier', 1.0)
    sm.params.set('learning_rate', 0.01)
    sm.params.set('random_exploratory_strategy.epsilon.decay.rate.min', 0.3)
    sm.params.set('random_exploratory_strategy.epsilon.decay.rate.initial', 0.999999)
    sm.params.set('reliability_threshold', 0.9)

    sm.params.get('features').remove(SupportedFeature.COMPOSITE_ACTIONS)

    return sm


def parse_args():
    """ Parses command line arguments.
    :return: argparse parser with parsed command line args
    """
    parser = argparse.ArgumentParser(description='Wumpus World Example Environment (Schema Mechanism)')

    parser.add_argument('--steps', type=int, required=False, default=N_STEPS,
                        help=f'the maximum number of steps before terminating an episode (default: {N_STEPS})')
    parser.add_argument('--episodes', type=int, required=False, default=N_EPISODES,
                        help=f'the maximum number of episodes (default: {N_EPISODES})')

    return parser.parse_args()


def run() -> None:
    args = parse_args()

    # note: randomizing the start seems to be extremely important in episodic environments because the SchemaMechanism
    # will learn unintended correlations due to repeated exposure to the same state state. For example, that turning
    # left results in a northerly orientation, or that moving forward (without constraint) leads to a particular
    # position.
    env = WumpusWorldMDP(worldmap=world, agent=agent_spec, wumpus=None, randomized_start=True)
    sm = create_schema_mechanism(env)

    display_params()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    start_time = time()

    for episode in range(args.episodes):

        # Initialize the world
        state, is_terminal = env.reset()

        # While not terminal state
        for step in range(args.steps):
            if is_terminal or not running:
                break

            info(f'n items: {len(ItemPool())}')
            info(f'n schemas: {len(sm.schema_memory)}')

            env.render()

            info(f'selection state[{step}]: {state}')

            selection_details = sm.select(state)

            current_composite_schema = sm.schema_selection.pending_schema
            if current_composite_schema:
                info(f'active composite action schema: {current_composite_schema} ')

            terminated_composite_schemas = selection_details.terminated_pending

            if terminated_composite_schemas:
                info(f'terminated schemas:')
                i = 1
                for pending_details in terminated_composite_schemas:
                    info(f'schema [{i}]: {pending_details.schema}')
                    info(f'selection state [{i}]: {pending_details.selection_state}')
                    info(f'status [{i}]: {pending_details.status}')
                    info(f'duration [{i}]: {pending_details.duration}')
                i += 1

            schema = selection_details.selected
            info(f'selected schema[{step}]: {schema}')

            state, is_terminal = env.step(schema.action)

            sm.learn(selection_details, result_state=state)

            if pause:
                display_item_values()
                # display_schema_info(sym_schema('/MOVE[FORWARD]/'))

                try:
                    while pause:
                        sleep(0.1)
                except KeyboardInterrupt:
                    pass

        GlobalStats().delegated_value_helper.eligibility_trace.clear()

    end_time = time()

    info(f'terminating execution after {end_time - start_time} seconds')

    display_summary(sm)

    # TODO: Add code to save a serialized instance of this schema mechanism to disk


if __name__ == "__main__":
    # TODO: Add code to load a serialized instance that was saved to disk

    run()
