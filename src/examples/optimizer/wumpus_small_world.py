import logging
import statistics
import traceback
from copy import copy
from dataclasses import dataclass
from typing import Callable

import optuna
from optuna import Trial

from schema_mechanism.core import default_global_params
from schema_mechanism.func_api import sym_item
from schema_mechanism.modules import init
from schema_mechanism.strategies.evaluation import DefaultEvaluationStrategy

logger = logging.getLogger(__name__)


def init_optimizer_trial(env, trial: Trial):
    logger.info('\thyper-parameters:')
    sampled_global_params = sample_global_params(trial)
    for param, value in sampled_global_params.items():
        logger.info(f'\t\t{param}:{value}')

    sampled_strategy_params = sample_strategy_params(trial)
    for param, value in sampled_strategy_params.items():
        logger.info(f'\t\t{param}:{value}')

    primitive_items = [
        sym_item('EVENT[AGENT ESCAPED]', primitive_value=0.75),
        sym_item('AGENT.HAS[GOLD]', primitive_value=0.25),
    ]

    # override default global parameters with sampled global parameters for this trial
    global_params = copy(default_global_params)
    for param, value in sampled_global_params.items():
        global_params.set(param, value)

    schema_mechanism = init(
        items=primitive_items,
        actions=env.actions,
        global_params=global_params
    )

    # override default evaluation strategy parameters with sampled strategy parameters for this trial
    schema_mechanism.schema_selection.evaluation_strategy = DefaultEvaluationStrategy(**sampled_strategy_params)

    return schema_mechanism


@dataclass
class EpisodeSummary:
    steps_taken: int = 0
    agent_escaped: bool = False
    agent_dead: bool = False
    wumpus_dead: bool = False
    gold_in_possession: int = 0
    arrows_in_possession: int = 0


def calculate_score(episode_summary: EpisodeSummary) -> float:
    # TODO: add the following into the score
    #    agent has gold
    #    wumpus dead
    #    agent dead
    weight_steps = 0.75
    weight_gold = 0.25

    return weight_steps * (1.0 / episode_summary.steps_taken) + weight_gold * episode_summary.gold_in_possession


# the objective function called by optimizer during each trial
def get_objective_function(env,
                           run: Callable,
                           n_runs_per_trial: int,
                           n_episodes_per_run: int,
                           n_steps_per_episode: int) -> Callable:
    def objective(trial: Trial):
        logger.info(f'*** beginning trial {trial.number}')

        try:
            # begin trial
            scores = []
            for run_id in range(n_runs_per_trial):
                schema_mechanism = init_optimizer_trial(env, trial)
                episode_summaries = run(
                    env,
                    schema_mechanism,
                    max_steps=n_steps_per_episode,
                    max_episodes=n_episodes_per_run,
                )

                score = statistics.mean([calculate_score(summary) for summary in episode_summaries])
                scores.append(score)

                logger.info(f'\trun {run_id}\'s score {score}')

            average_score = statistics.mean(scores)
            median_score = statistics.median(scores)
            std_dev_score = statistics.stdev(scores)

            logger.info(f'trial number {trial.number}\'s results')

            logger.info(f'\taverage score: {average_score}')
            logger.info(f'\tmedian score: {median_score}')
            logger.info(f'\tstd_dev score: {std_dev_score}')

            # optuna minimizes the objective by default, so we need to flip the sign to maximize
            cost = -1 * average_score

            return cost

        except (AssertionError, ValueError) as e:
            logger.error(f'pruning optimizer trial {trial} due to exception {e}')
            tb = traceback.format_exc()
            logger.error(tb)

            raise optuna.exceptions.TrialPruned()

    return objective


def sample_global_params(trial: Trial) -> dict:
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1)
    backward_chains_max_length = trial.suggest_int(
        'composite_actions.backward_chains.max_length',
        low=1,
        high=5,
        step=1
    )
    min_baseline_advantage = trial.suggest_float(
        'composite_actions.min_baseline_advantage',
        low=0.05,
        high=1.0,
        step=0.05
    )
    schema_reliability_threshold = trial.suggest_float(
        'schema.reliability_threshold',
        low=0.6,
        high=1.0,
        step=0.1
    )

    correlation_tests = [
        # 'BarnardExactCorrelationTest',
        'CorrelationOnEncounter',
        'DrescherCorrelationTest',
        'FisherExactCorrelationTest',
    ]

    ext_context_correlation_test = trial.suggest_categorical(
        'ext_context.correlation_test',
        choices=correlation_tests
    )

    ext_result_correlation_test = trial.suggest_categorical(
        'ext_result.correlation_test',
        choices=correlation_tests
    )

    ext_context_positive_correlation_threshold = trial.suggest_float(
        'ext_context.positive_correlation_threshold',
        low=0.55,
        high=1.0,
        step=0.05
    )

    ext_result_positive_correlation_threshold = trial.suggest_float(
        'ext_result.positive_correlation_threshold',
        low=0.55,
        high=1.0,
        step=0.05
    )

    return {
        'learning_rate': learning_rate,
        'composite_actions.backward_chains.max_length': backward_chains_max_length,
        'composite_actions.min_baseline_advantage': min_baseline_advantage,
        'schema.reliability_threshold': schema_reliability_threshold,
        'ext_context.correlation_test': ext_context_correlation_test,
        'ext_result.correlation_test': ext_result_correlation_test,
        'ext_context.positive_correlation_threshold': ext_context_positive_correlation_threshold,
        'ext_result.positive_correlation_threshold': ext_result_positive_correlation_threshold,
    }


def sample_strategy_params(trial: Trial) -> dict:
    epsilon = trial.suggest_categorical(
        'epsilon',
        choices=[0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999],
    )

    epsilon_min = trial.suggest_float(
        'epsilon_min',
        low=0.0,
        high=0.3,
        step=0.025,
    )

    epsilon_decay_rate = trial.suggest_categorical(
        'epsilon_decay_rate',
        choices=[0.9, 0.99, 0.999, 0.9999, 0.99999],
    )

    reliability_max_penalty = trial.suggest_float(
        'reliability_max_penalty',
        low=0.1,
        high=1.0,
        step=0.1,
    )

    pending_focus_max_value = trial.suggest_float(
        'pending_focus_max_value',
        low=0.1,
        high=1.0,
        step=0.1,
    )

    pending_focus_decay_rate = trial.suggest_float(
        'pending_focus_decay_rate',
        low=0.1,
        high=0.9,
        step=0.1,
    )

    return {
        'epsilon': epsilon,
        'epsilon_min': epsilon_min,
        'epsilon_decay_rate': epsilon_decay_rate,
        'reliability_max_penalty': reliability_max_penalty,
        'pending_focus_max_value': pending_focus_max_value,
        'pending_focus_decay_rate': pending_focus_decay_rate,
    }
