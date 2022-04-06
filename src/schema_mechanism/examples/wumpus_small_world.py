from time import time

from schema_mechanism.examples import display_summary
from schema_mechanism.examples.environments.wumpus_world import Wumpus
from schema_mechanism.examples.environments.wumpus_world import WumpusWorldAgent
from schema_mechanism.examples.environments.wumpus_world import WumpusWorldMDP
from schema_mechanism.func_api import sym_item
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import info
# Wumpus @ (6,3); Agent starts @ (1,8) facing 'W'
from schema_mechanism.stats import FisherExactCorrelationTest

agent_spec = WumpusWorldAgent(position=(1, 8), direction='W', n_arrows=0)
wumpus_spec = Wumpus(position=(6, 3), health=2)
world = """
wwwwwwwwww
w........w 
w.wwa.wwww
w.ppwwwppw
w.pwww..ew
w.pgpw.ppw
w...w..www
w..ww.wwpw
w.......aw
wwwwwwwwww
"""

# global constants
N_EPISODES = 5000

if __name__ == "__main__":

    GlobalParams().set('learning_rate', 0.25)
    GlobalParams().set('reliability_threshold', 0.8)
    GlobalParams().set('habituation_decay_rate', 0.95)
    GlobalParams().set('habituation_multiplier', 100.0)
    GlobalParams().set('max_reliability_penalty', 10.0)
    GlobalParams().set('goal_weight', 0.1)
    GlobalParams().set('explore_weight', 0.9)
    GlobalParams().set('dv_discount_factor', 0.9)
    GlobalParams().set('dv_decay_rate', 0.3)

    env = WumpusWorldMDP(world, agent_spec, wumpus_spec)

    n_episodes = 0

    item_wumpus_wounded = sym_item('EVENT[WUMPUS WOUNDED]', primitive_value=250.0)
    item_wumpus_dead = sym_item('EVENT[WUMPUS DEAD]', primitive_value=500.0)
    item_agent_escaped = sym_item('EVENT[AGENT ESCAPED]', primitive_value=1000.0)
    item_agent_dead = sym_item('AGENT.HEALTH=0', primitive_value=-10000.0)

    items_has_gold = []
    for n_gold in range(0, 5):
        items_has_gold.append(sym_item(f'AGENT.HAS[GOLD:{n_gold}]', primitive_value=n_gold * 100.0))

    items_has_arrows = []
    for n_arrows in range(0, 5):
        items_has_arrows.append(sym_item(f'AGENT.HAS[ARROWS:{n_arrows}]', primitive_value=n_arrows * 10.0))

    items = [
        item_wumpus_dead,
        item_wumpus_wounded,
        item_agent_dead,
        item_agent_escaped,
        *items_has_gold,
        *items_has_arrows
    ]

    sm = SchemaMechanism(primitive_actions=env.actions, primitive_items=items)

    start_time = time()

    while n_episodes < N_EPISODES:

        # Initialize the world
        state, is_terminal = env.reset()

        # While not terminal state
        n = 0
        while not is_terminal:
            env.render()

            info(f'state[{n}]: {state}')

            # TODO: action should be chosen using the SchemaMechanism
            schema = sm.select(state)
            info(f'selected schema[{n}]: {schema}')

            action = schema.action
            info(f'action[{n}]: {action}')

            # Observe next state and reward
            state, is_terminal = env.step(action)

            n += 1

            print(f'n_schemas: {len(sm.schema_memory)}')
            print(
                f'cache info (positive_corr_statistic): {FisherExactCorrelationTest.positive_corr_statistic.cache_info()}')
            print(
                f'cache info (negative_corr_statistic): {FisherExactCorrelationTest.negative_corr_statistic.cache_info()}')

    end_time = time()

    info(f'elapsed time: {end_time - start_time}s')

    display_summary(sm)
