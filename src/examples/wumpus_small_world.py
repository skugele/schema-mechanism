from time import sleep
from time import time

from pynput import keyboard

from examples import display_schema_info
from examples import display_summary
from examples.environments.wumpus_world import Wumpus
from examples.environments.wumpus_world import WumpusWorldAgent
from examples.environments.wumpus_world import WumpusWorldMDP
from schema_mechanism.core import GlobalStats
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import SupportedFeature
from schema_mechanism.share import display_params
from schema_mechanism.share import info
# Wumpus @ (6,3); Agent starts @ (1,8) facing 'W'
from schema_mechanism.stats import CorrelationOnEncounter
from schema_mechanism.stats import DrescherCorrelationTest

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
N_EPISODES = 5000

pause = False
running = True


def on_press(key):
    global pause, running

    if key == keyboard.Key.space:
        pause = not pause
    if key == keyboard.Key.esc:
        print('setting running to False', flush=True)
        running = False


def run(sm: SchemaMechanism) -> None:
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    start_time = time()
    while running:

        # Initialize the world
        state, is_terminal = env.reset()

        # While not terminal state
        n = 0
        while not is_terminal and running and n < 250:
            env.render()

            info(f'selection state[{n}]: {state}')

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
            info(f'selected schema[{n}]: {schema}')

            state, is_terminal = env.step(schema.action)

            sm.learn(selection_details, result_state=state)

            n += 1

            if pause:
                display_summary(sm)
                display_schema_info(sym_schema('/MOVE[FORWARD]/'))

                try:
                    while pause:
                        sleep(0.1)
                except KeyboardInterrupt:
                    pass

        GlobalStats().delegated_value_helper.eligibility_trace.clear()

    end_time = time()

    info(f'terminating execution after {end_time - start_time} seconds')


def set_params():
    GlobalParams().set('composite_action_min_baseline_advantage', 0.0)
    GlobalParams().set('delegated_value_helper.decay_rate', 0.5)
    GlobalParams().set('delegated_value_helper.discount_factor', 0.7)
    GlobalParams().set('random_exploratory_strategy.epsilon.decay.rate', 0.9999)
    GlobalParams().set('random_exploratory_strategy.epsilon.decay.min', 0.2)
    GlobalParams().set('schema_selection.weights.explore_weight', 0.2)
    GlobalParams().set('schema_selection.weights.goal_weight', 0.8)
    GlobalParams().set('habituation_exploratory_strategy.decay.rate', 0.95)
    GlobalParams().set('habituation_exploratory_strategy.multiplier', 5.0)
    GlobalParams().set('learning_rate', 0.005)
    GlobalParams().set('goal_pursuit_strategy.reliability.max_penalty', 5.0)
    GlobalParams().set('reliability_threshold', 0.9)
    GlobalParams().set('backward_chains.max_len', 2)
    GlobalParams().set('backward_chains.update_frequency', 0.001)

    GlobalParams().get('features').remove(SupportedFeature.COMPOSITE_ACTIONS)

    # item correlation test used for determining relevance of extended context items
    GlobalParams().set('ext_context.correlation_test', DrescherCorrelationTest)

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    GlobalParams().set('ext_context.positive_correlation_threshold', 0.95)
    GlobalParams().set('ext_context.negative_correlation_threshold', 1.0)

    # item correlation test used for determining relevance of extended result items
    GlobalParams().set('ext_result.correlation_test', CorrelationOnEncounter)

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    GlobalParams().set('ext_result.positive_correlation_threshold', 0.95)
    GlobalParams().set('ext_result.negative_correlation_threshold', 0.95)


if __name__ == "__main__":
    set_params()
    display_params()

    # env = WumpusWorldMDP(world, agent_spec, wumpus_spec)

    # note: randomizing the start seems to be extremely important in episodic environments because the SchemaMechanism
    # will learn unintended correlations due to repeated exposure to the same state state. For example, that turning
    # left results in a northerly orientation, or that moving forward (without constraint) leads to a particular
    # position.
    env = WumpusWorldMDP(worldmap=world, agent=agent_spec, wumpus=None, randomized_start=True)

    n_episodes = 0

    # item_wumpus_wounded = sym_item('EVENT[WUMPUS WOUNDED]', primitive_value=250.0)
    # item_wumpus_dead = sym_item('EVENT[WUMPUS DEAD]', primitive_value=500.0)
    item_agent_escaped = sym_item('EVENT[AGENT ESCAPED]', primitive_value=100.0)
    # item_agent_dead = sym_item('EVENT[AGENT DEAD]', primitive_value=-10000.0)
    # item_agent_dead = sym_item('AGENT.HEALTH=0', primitive_value=-10000.0)
    # items_has_gold = (sym_item(f'AGENT.HAS[GOLD]', primitive_value=500.0))

    items = [
        # item_agent_dead,
        item_agent_escaped,
        # items_has_gold,
    ]

    sm = SchemaMechanism(primitive_actions=env.actions, primitive_items=items)

    run(sm)
