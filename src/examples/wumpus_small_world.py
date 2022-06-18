import argparse
import logging.config
from pathlib import Path
from statistics import mean
from time import sleep
from time import time
from typing import Collection
from typing import Optional

import optuna
from optuna.integration import SkoptSampler
from optuna.pruners import MedianPruner
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from pandas import DataFrame

from examples import RANDOM_SEED
from examples import display_item_values
from examples import display_known_schemas
from examples import is_paused
from examples import is_running
from examples import parse_optimizer_args
from examples import parser_serialization_args
from examples import run_decorator
from examples.environments.wumpus_world import WumpusWorldAgent
from examples.environments.wumpus_world import WumpusWorldMDP
from examples.optimizer.wumpus_small_world import EpisodeSummary
from examples.optimizer.wumpus_small_world import get_objective_function
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import sym_item
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import init
from schema_mechanism.modules import load
from schema_mechanism.modules import save
from schema_mechanism.parameters import GlobalParams
from schema_mechanism.util import set_random_seed

logger = logging.getLogger('examples.environments.wumpus_small_world')

# For reproducibility, we we also need to set PYTHONHASHSEED=RANDOM_SEED in the environment
set_random_seed(RANDOM_SEED)

MAX_EPISODES = 50
MAX_STEPS = 25


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


def parse_env_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--steps', type=int, required=False, default=MAX_STEPS,
                        help=f'the maximum number of steps before terminating an episode (default: {MAX_STEPS})')
    parser.add_argument('--episodes', type=int, required=False, default=MAX_EPISODES,
                        help=f'the maximum number of episodes (default: {MAX_EPISODES})')
    parser.add_argument('--render_env', required=False, action="store_true", default=False,
                        help=f'display an ASCII rendering of the environment (default: {False})')

    return parser


def parse_args():
    """ Parses command line arguments.
    :return: argparse parser with parsed command line args
    """
    parser = argparse.ArgumentParser(description='Wumpus World Example Environment (Schema Mechanism)')

    parser = parse_env_args(parser)
    parser = parse_optimizer_args(parser)
    parser = parser_serialization_args(parser)

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
def run(env: WumpusWorldMDP,
        schema_mechanism: SchemaMechanism,
        max_steps: int,
        max_episodes: int,
        render_env: bool = False) -> Collection[EpisodeSummary]:
    env.reset()

    episode_summaries = list()
    for episode in range(1, max_episodes + 1):
        episode_summary: EpisodeSummary = EpisodeSummary()

        # initialize the world
        selection_state, is_terminal = env.reset()

        for step in range(1, max_steps + 1):
            if is_terminal or not is_running():
                break

            progress_id = f'{episode}:{step}'

            if render_env:
                logger.debug(f'\n{env.render()}\n')

            logger.debug(f'state [{progress_id}]: {selection_state}')
            selection_details = schema_mechanism.select(selection_state)

            current_composite_schema = schema_mechanism.schema_selection.pending_schema
            if current_composite_schema:
                logger.debug(f'active composite action schema [{progress_id}]: {current_composite_schema} ')

            terminated_composite_schemas = selection_details.terminated_pending
            if terminated_composite_schemas:
                logger.debug(f'terminated schemas:')
                for i, pending_details in enumerate(terminated_composite_schemas):
                    logger.debug(f'schema [{i}]: {pending_details.schema}')
                    logger.debug(f'selection state [{i}]: {pending_details.selection_state}')
                    logger.debug(f'status [{i}]: {pending_details.status}')
                    logger.debug(f'duration [{i}]: {pending_details.duration}')

            schema = selection_details.selected
            action = schema.action
            effective_value = selection_details.effective_value

            logger.debug(f'selected schema [{progress_id}]: {schema} [eff. value: {effective_value}]')

            result_state, is_terminal = env.step(action)

            schema_mechanism.learn(selection_details, result_state=result_state)

            if is_paused():
                display_item_values()
                display_known_schemas(schema_mechanism)

                # display_schema_info(sym_schema(f'/USE[EXIT]/'))
                if episode > 1:
                    display_performance_summary(episode_summaries)

                try:
                    while is_paused():
                        sleep(0.1)
                except KeyboardInterrupt:
                    pass

            selection_state = result_state

            # update episode summary
            episode_summary.steps_taken = step
            episode_summary.agent_escaped = env.agent.has_escaped
            episode_summary.agent_dead = env.agent.health
            episode_summary.wumpus_dead = env.wumpus.health if env.wumpus else 0
            episode_summary.gold_in_possession = env.agent.n_gold
            episode_summary.arrows_in_possession = env.agent.n_arrows

        episode_summaries.append(episode_summary)

        # GlobalStats().delegated_value_helper.eligibility_trace.clear()

    return episode_summaries


def display_performance_summary(episode_summaries: Collection[EpisodeSummary]) -> None:
    steps_in_episode = [episode_summary.steps_taken for episode_summary in episode_summaries]

    steps_in_last_episode = steps_in_episode[-1]
    n_episodes = len(episode_summaries)
    n_episodes_agent_escaped = sum(1 for episode_summary in episode_summaries if episode_summary.agent_escaped)
    n_episodes_agent_has_gold = sum(1 for episode_summary in episode_summaries if episode_summary.gold_in_possession)
    avg_steps_per_episode = mean(steps_in_episode)
    min_steps_per_episode = min(steps_in_episode)
    max_steps_per_episode = max(steps_in_episode)
    total_steps = sum(steps_in_episode)

    logger.info(f'\tepisodes: {n_episodes}')
    logger.info(f'\tepisodes in which agent escaped: {n_episodes_agent_escaped}')
    logger.info(f'\tepisodes in which agent had gold: {n_episodes_agent_has_gold}')
    logger.info(f'\tcumulative steps: {total_steps}')
    logger.info(f'\tsteps in last episode: {steps_in_last_episode}')
    logger.info(f'\taverage steps per episode: {avg_steps_per_episode}')
    logger.info(f'\tminimum steps per episode: {min_steps_per_episode}')
    logger.info(f'\tmaximum steps per episode: {max_steps_per_episode}')


def optimize(env: WumpusWorldMDP,
             sampler: str,
             pruner: str,
             n_trials: int,
             n_runs_per_trial: int,
             n_episodes_per_run: int,
             n_steps_per_episode: int,
             optimizer_study_name: str = None,
             optimizer_use_db: bool = False
             ) -> DataFrame:
    seed = int(time())

    if sampler == 'random':
        sampler = RandomSampler(seed=seed)
    elif sampler == 'tpe':
        sampler = TPESampler(n_startup_trials=5, seed=seed)
    elif sampler == 'skopt':
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})
    else:
        raise ValueError('Unknown sampler: {}'.format(sampler))

    if pruner == 'halving':
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner == 'median':
        pruner = MedianPruner(n_startup_trials=5)
    elif pruner == 'none':
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials)
    else:
        raise ValueError('Unknown pruner: {}'.format(pruner))

    study_name = optimizer_study_name or f'wumpus-small-world-{int(time())}-optimizer_study'
    storage = f'sqlite:///{study_name}.db'

    study = optuna.create_study(study_name=study_name,
                                storage=storage if optimizer_use_db else None,
                                load_if_exists=True,
                                sampler=sampler,
                                pruner=pruner)

    try:
        objective = get_objective_function(
            env=env,
            run=run,
            n_runs_per_trial=n_runs_per_trial,
            n_episodes_per_run=n_episodes_per_run,
            n_steps_per_episode=n_steps_per_episode,
        )
        study.optimize(objective, n_trials)
    except KeyboardInterrupt:
        pass

    logger.info(f'Number of finished trials: {len(study.trials)}')
    logger.info('Best trial:')
    best_trial = study.best_trial

    logger.info(f'\tValue: {-best_trial.value}')
    logger.info('\tParams: ')
    for key, value in best_trial.params.items():
        logger.info(f'\t\t{key}: {value}')

    return study.trials_dataframe()


def load_schema_mechanism(model_filepath: Path) -> SchemaMechanism:
    schema_mechanism: Optional[SchemaMechanism] = None
    try:
        schema_mechanism = load(model_filepath)
    except FileNotFoundError as e:
        logger.error(e)

    return schema_mechanism


def create_schema_mechanism(env: WumpusWorldMDP) -> SchemaMechanism:
    primitive_items = [
        sym_item('EVENT[AGENT ESCAPED]', primitive_value=1.0),
        # sym_item('(EVENT[AGENT ESCAPED],AGENT.HAS[GOLD])', primitive_value=1.0),
    ]

    schema_mechanism = init(
        items=primitive_items,
        actions=env.actions,
        global_params=init_params()
    )

    return schema_mechanism


def main():
    # configure logger
    logging.config.fileConfig('config/logging.conf')

    args = parse_args()

    env = create_world()

    if args.optimize:
        data_frame = optimize(
            env=env,
            sampler=args.optimizer_sampler,
            pruner=args.optimizer_pruner,
            n_trials=args.optimizer_trials,
            n_runs_per_trial=args.optimizer_runs_per_trial,
            n_episodes_per_run=args.episodes,
            n_steps_per_episode=args.steps,
            optimizer_study_name=args.optimizer_study_name,
            optimizer_use_db=args.optimizer_use_db,
        )

        report_name = f'optimizer_results_{int(time())}-{args.optimizer_sampler}-{args.optimizer_pruner}.csv'
        report_path = f'local/optimize/wumpus_small_world/{report_name}'
        logger.info("Writing report to {}".format(report_path))

        data_frame.to_csv(report_path)
    else:
        load_path: Path = args.load_path
        schema_mechanism = (
            load_schema_mechanism(load_path)
            if load_path
            else create_schema_mechanism(env)
        )

        episode_summaries = run(
            env=env,
            schema_mechanism=schema_mechanism,
            max_steps=args.steps,
            max_episodes=args.episodes,
            render_env=args.render_env,
        )

        display_performance_summary(episode_summaries)

        save_path: Path = args.save_path
        if save_path:
            save(
                modules=[schema_mechanism],
                path=save_path,
            )


if __name__ == '__main__':
    main()
