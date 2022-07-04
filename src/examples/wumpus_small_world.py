import argparse
import logging.config
import typing
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Collection
from typing import Iterable
from typing import Optional

from examples import RANDOM_SEED
from examples import Runner
from examples import SaveCallback
from examples import default_display_on_step
from examples import parse_optimizer_args
from examples import parse_run_args
from examples import parser_serialization_args
from examples import run_episode
from examples import save_every_n_steps
from examples.environments.wumpus_world import WumpusWorldAgent
from examples.environments.wumpus_world import WumpusWorldMdp
from examples.environments.wumpus_world import WumpusWorldMdpEpisodeSummary
from examples.optimizers import get_optimizer_report_filename
from examples.optimizers import optimize
from schema_mechanism.core import Item
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import sym_item
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import init
from schema_mechanism.modules import load
from schema_mechanism.modules import save
from schema_mechanism.parameters import GlobalParams
from schema_mechanism.serialization import DEFAULT_ENCODING
from schema_mechanism.strategies.evaluation import DefaultEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGreedyEvaluationStrategy
from schema_mechanism.util import set_random_seed

logger = logging.getLogger('examples.environments.wumpus_small_world')

# For reproducibility, we we also need to set PYTHONHASHSEED=RANDOM_SEED in the environment
set_random_seed(RANDOM_SEED)

MAX_EPISODES = 25
MAX_STEPS = 100


def init_params() -> GlobalParams:
    params = get_global_params()

    params.set('learning_rate', 0.1)

    params.set('schema.reliability_threshold', 0.9)

    params.set('composite_actions.backward_chains.max_length', 3)
    params.set('composite_actions.update_frequency', 0.005)
    params.set('composite_actions.min_baseline_advantage', 0.4)

    params.set('ext_context.correlation_test', 'DrescherCorrelationTest')
    params.set('ext_context.positive_correlation_threshold', 0.8)
    params.set('ext_result.correlation_test', 'CorrelationOnEncounter')
    params.set('ext_result.positive_correlation_threshold', 0.8)

    return params


# TODO: update this with parameters that will influence world creation
def parse_env_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


def parse_args():
    """ Parses command line arguments.
    :return: argparse parser with parsed command line args
    """
    parser = argparse.ArgumentParser(description='Wumpus World Example Environment (Schema Mechanism)')

    parser = parse_run_args(parser)
    parser = parse_env_args(parser)
    parser = parse_optimizer_args(parser)
    parser = parser_serialization_args(parser)

    return parser.parse_args()


def create_world() -> WumpusWorldMdp:
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
    env = WumpusWorldMdp(
        world_id='wumpus-small-world',
        world_map=world,
        agent=agent_spec,
        wumpus=None,
        randomized_start=True
    )

    return env


def calculate_score(episode_summary: WumpusWorldMdpEpisodeSummary) -> float:
    # TODO: add the following into the score
    #    agent has gold
    #    wumpus dead
    #    agent dead
    weight_steps = 0.75
    weight_gold = 0.25

    return weight_steps * (1.0 / episode_summary.steps) + weight_gold * episode_summary.gold


def display_performance_summary(episode_summaries: Collection[WumpusWorldMdpEpisodeSummary]) -> None:
    logger.info(f'Performance Summary:')

    steps_in_episode = [episode_summary.steps for episode_summary in episode_summaries]

    steps_in_last_episode = steps_in_episode[-1]
    n_episodes = len(episode_summaries)
    n_episodes_agent_escaped = sum(1 for episode_summary in episode_summaries if episode_summary.agent_escaped)
    n_episodes_agent_has_gold = sum(episode_summary.gold for episode_summary in episode_summaries)
    n_episodes_agent_escaped_and_has_gold = sum(
        1
        for episode_summary in episode_summaries
        if episode_summary.gold and episode_summary.agent_escaped
    )

    avg_steps_per_episode = mean(steps_in_episode)
    min_steps_per_episode = min(steps_in_episode)
    max_steps_per_episode = max(steps_in_episode)
    cumulative_steps = sum(steps_in_episode)

    logger.info(f'\tEpisodes: {n_episodes}')
    logger.info(f'\tEpisodes in which agent escaped: {n_episodes_agent_escaped}')
    logger.info(f'\tEpisodes in which agent had gold: {n_episodes_agent_has_gold}')
    logger.info(f'\tEpisodes in which agent escaped AND had gold: {n_episodes_agent_escaped_and_has_gold}')
    logger.info(f'\tCumulative steps: {cumulative_steps}')
    logger.info(f'\tSteps in last episode: {steps_in_last_episode}')
    logger.info(f'\tAverage steps per episode: {avg_steps_per_episode}')
    logger.info(f'\tMinimum steps per episode: {min_steps_per_episode}')
    logger.info(f'\tMaximum steps per episode: {max_steps_per_episode}')


def load_schema_mechanism(model_filepath: Path) -> SchemaMechanism:
    schema_mechanism: Optional[SchemaMechanism] = None
    try:
        schema_mechanism = load(model_filepath)
    except FileNotFoundError as e:
        logger.error(e)

    return schema_mechanism


def create_schema_mechanism(env: WumpusWorldMdp, primitive_items: Iterable[Item]) -> SchemaMechanism:
    schema_mechanism = init(
        items=primitive_items,
        actions=env.actions,
        global_params=init_params()
    )

    schema_mechanism.schema_selection.evaluation_strategy = DefaultEvaluationStrategy(
        goal_pursuit_strategy=DefaultGoalPursuitEvaluationStrategy(
            reliability_max_penalty=0.6,
            pending_focus_max_value=0.9,
            pending_focus_decay_rate=0.4
        ),
        exploratory_strategy=DefaultExploratoryEvaluationStrategy(
            epsilon=0.9999,
            epsilon_min=0.025,
            epsilon_decay_rate=0.9999,
        ),
    )

    return schema_mechanism


def main():
    # configure logger
    logging.config.fileConfig('config/logging.conf')

    # parse command line arguments
    args = parse_args()

    load_path: Path = args.load_path
    save_path: Path = args.save_path

    # initialize environment
    env = create_world()

    primitive_items = [
        sym_item('EVENT[AGENT ESCAPED]', primitive_value=1.0),
        sym_item('(EVENT[AGENT ESCAPED],AGENT.HAS[GOLD])', primitive_value=1.0),
    ]

    if args.optimize:
        # this typing cast is a workaround to a current limitation in type inference for partials that affects PyCharm
        # and likely other IDEs/static code analysis tools.
        run: Runner = typing.cast(Runner, partial(run_episode, on_step=None))

        data_frame = optimize(
            env=env,
            primitive_items=primitive_items,
            run_episode=run,
            calculate_score=calculate_score,
            sampler=args.optimizer_sampler,
            pruner=args.optimizer_pruner,
            n_trials=args.optimizer_trials,
            n_runs_per_trial=args.optimizer_runs_per_trial,
            n_steps_per_episode=args.steps,
            n_episodes_per_run=args.episodes,
            study_name=args.optimizer_study_name,
            use_database=args.optimizer_use_db,
        )

        report_filename = get_optimizer_report_filename(
            study_name=args.optimizer_study_name,
            sampler=args.optimizer_sampler,
            pruner=args.optimizer_pruner,
        )

        report_path = f'local/optimize/{env.id}/{report_filename}'
        logger.info("Writing report to {}".format(report_path))

        data_frame.to_csv(report_path)
    else:
        on_step_callbacks = [default_display_on_step]
        if save_path:
            on_step_callbacks.append(
                SaveCallback(
                    path=save_path,
                    conditions=[save_every_n_steps(args.save_frequency)])
            )

        # this typing cast is a workaround to a current limitation in type inference for partials that affects PyCharm
        # and likely other IDEs/static code analysis tools.
        run: Runner = typing.cast(Runner, partial(run_episode, on_step=on_step_callbacks))

        schema_mechanism = load_schema_mechanism(load_path) if load_path else None
        if not schema_mechanism:
            schema_mechanism = create_schema_mechanism(
                env=env,
                primitive_items=primitive_items,
            )

        # command-line argument overrides schema mechanism's strategy parameters
        if args.greedy:
            schema_mechanism.schema_selection.evaluation_strategy = DefaultGreedyEvaluationStrategy(
                reliability_threshold=get_global_params().get('schema.reliability_threshold')
            )

        episode_summaries = []

        # episode loop
        for episode_id in range(args.episodes):
            logger.debug(f'starting episode {episode_id}')

            # execute a single episode
            episode_summary = run(
                env=env,
                schema_mechanism=schema_mechanism,
                max_steps=args.steps,
                episode=episode_id,
                render_env=False
            )

            episode_summaries.append(episode_summary)

            # displays a summary of partial results
            display_performance_summary(episode_summaries)

        # this save is in addition to any runner callbacks to make sure final model is persisted
        if save_path:
            save(
                modules=[schema_mechanism],
                path=save_path,
                encoding=DEFAULT_ENCODING,
            )


if __name__ == '__main__':
    main()
