import logging
import statistics
import traceback
from copy import copy
from time import time
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional

import numpy as np
import optuna
from optuna import Trial
from optuna.integration import SkoptSampler
from optuna.pruners import MedianPruner
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from pandas import DataFrame

from examples import Runner
from examples.environments import Environment
from schema_mechanism.core import Item
from schema_mechanism.core import default_global_params
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import init
from schema_mechanism.strategies.evaluation import DefaultEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.weight_update import CyclicWeightUpdateStrategy

logger = logging.getLogger(__name__)


def display_hyper_parameters(trial: Trial) -> None:
    logger.info(f'{"-" * 80}')
    for param, value in trial.params.items():
        logger.info(f'\t{param}:{value}')
    logger.info(f'{"-" * 80}')


def get_optimizer_report_filename(
        prefix: str = 'optimizer_results',
        study_name: str = None,
        sampler: Optional[str] = None,
        pruner: Optional[str] = None,
        add_timestamp: bool = True) -> str:
    name_components = [component for component in [prefix, study_name, sampler, pruner] if component]
    if add_timestamp:
        name_components.append(str(int(time())))

    report_name = '-'.join(name_components)
    if not report_name:
        raise ValueError('Report name cannot be an empty string')

    return f'{report_name}.csv'


def sample_global_params(trial: Trial) -> dict:
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    backward_chains_max_length = trial.suggest_int(
        'composite_actions.backward_chains.max_length',
        low=2,
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
        choices=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999],
    )

    epsilon_min = trial.suggest_float(
        'epsilon_min',
        low=0.0,
        high=0.5,
        step=0.025,
    )

    epsilon_decay_rate = trial.suggest_categorical(
        'epsilon_decay_rate',
        choices=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999],
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

    weight_update_step_size = trial.suggest_float(
        'weight_update_step_size',
        low=1e-6,
        high=1e-1,
    )

    initial_exploratory_weight = trial.suggest_float(
        'initial_exploratory_weight',
        low=0.0,
        high=1.0,
        step=0.1,
    )

    initial_exploratory_weight = np.round(initial_exploratory_weight, decimals=1)
    initial_goal_pursuit_weight = np.round(1.0 - initial_exploratory_weight, decimals=1)

    return {
        'epsilon': epsilon,
        'epsilon_min': epsilon_min,
        'epsilon_decay_rate': epsilon_decay_rate,
        'reliability_max_penalty': reliability_max_penalty,
        'pending_focus_max_value': pending_focus_max_value,
        'pending_focus_decay_rate': pending_focus_decay_rate,
        'weight_update_step_size': weight_update_step_size,
        'initial_exploratory_weight': initial_exploratory_weight,
        'initial_goal_pursuit_weight': initial_goal_pursuit_weight,
    }


def init_optimizer_trial(env: Environment,
                         primitive_items: Iterable[Item],
                         trial_global_params: dict,
                         trial_strategy_params: dict) -> SchemaMechanism:
    # override default global parameters with sampled global parameters for this trial
    global_params = copy(default_global_params)
    for param, value in trial_global_params.items():
        global_params.set(param, value)

    schema_mechanism = init(
        items=primitive_items,
        actions=env.actions,
        global_params=global_params
    )

    # override default evaluation strategy parameters with sampled strategy parameters for this trial
    initial_weights = [
        trial_strategy_params['initial_goal_pursuit_weight'],
        trial_strategy_params['initial_exploratory_weight'],
    ]

    schema_mechanism.schema_selection.evaluation_strategy = DefaultEvaluationStrategy(
        weights=initial_weights,
        weight_update_strategy=CyclicWeightUpdateStrategy(
            step_size=trial_strategy_params.get('weight_update_step_size')
        )
    )

    schema_mechanism.schema_selection.evaluation_strategy = DefaultEvaluationStrategy(
        goal_pursuit_strategy=DefaultGoalPursuitEvaluationStrategy(
            reliability_max_penalty=trial_strategy_params.get('reliability_max_penalty'),
            pending_focus_max_value=trial_strategy_params.get('pending_focus_max_value'),
            pending_focus_decay_rate=trial_strategy_params.get('pending_focus_decay_rate'),
        ),
        exploratory_strategy=DefaultExploratoryEvaluationStrategy(
            epsilon=trial_strategy_params.get('epsilon'),
            epsilon_min=trial_strategy_params.get('epsilon_min'),
            epsilon_decay_rate=trial_strategy_params.get('epsilon_decay_rate'),
        ),
    )

    return schema_mechanism


def optimize(env: Environment,
             primitive_items: Iterable[Item],
             run_episode: Runner,
             calculate_score: Callable[[Any], float],
             sampler: str,
             pruner: str,
             n_trials: int,
             n_runs_per_trial: int,
             n_steps_per_episode: int,
             n_episodes_per_run: int = 1,
             study_name: Optional[str] = None,
             use_database: bool = False,
             show_progress_bar: bool = False,
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

    session_id = int(time())
    study_name = study_name if study_name is not None else f'{env.id}-{session_id}-optimizer_study'
    storage = (
        f'sqlite:///{study_name}.db'
        if use_database
        else None
    )
    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=True,
                                sampler=sampler,
                                pruner=pruner)

    try:
        objective = get_objective_function(
            env=env,
            primitive_items=primitive_items,
            run_episode=run_episode,
            calculate_score=calculate_score,
            n_runs_per_trial=n_runs_per_trial,
            n_episodes_per_run=n_episodes_per_run,
            n_steps_per_episode=n_steps_per_episode,
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
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


def get_objective_function(env: Environment,
                           primitive_items: Iterable[Item],
                           run_episode: Runner,
                           calculate_score: Callable[[Any], float],
                           n_steps_per_episode: int,
                           n_runs_per_trial: int = 1,
                           n_episodes_per_run: int = 1,
                           ) -> Callable:
    """

    :param env: the Environment for which the optimizer study will be run.
    :param primitive_items: an iterable containing the agent's primitive items
    :param run_episode: the callable that executes a trial's run's episode(s).
    :param calculate_score: a callable that calculates an episode's score based on result returned by run.
    :param n_steps_per_episode: the maximum number of agent steps allowed for an episode.
    :param n_runs_per_trial: the number of runs per optimizer trial.
    :param n_episodes_per_run: the number of episodes per run.

    :return: the objective function with respect to which parameters will be optimized (a Callable)
    """

    # The callable returned from this method. Used by Optuna as the optimizer's objective function. It is invoked
    # once for each trial.
    def objective(trial: Trial):
        logger.info(f'*** beginning trial {trial.number}')

        sampled_global_params = sample_global_params(trial)
        sampled_strategy_params = sample_strategy_params(trial)

        display_hyper_parameters(trial)

        try:
            scores = []

            for run_id in range(n_runs_per_trial):
                logger.info(f'*** beginning run {run_id} for trial {trial.number}')

                for episode_id in range(n_episodes_per_run):
                    schema_mechanism = init_optimizer_trial(
                        env,
                        primitive_items=primitive_items,
                        trial_global_params=sampled_global_params,
                        trial_strategy_params=sampled_strategy_params,
                    )

                    results = run_episode(
                        env=env,
                        schema_mechanism=schema_mechanism,
                        max_steps=n_steps_per_episode,
                        episode=episode_id,
                        render_env=False
                    )

                    score = calculate_score(results)
                    scores.append(score)

                    logger.debug(f'episode {episode_id} score: {score}')

            best_score = max(scores)
            average_score = statistics.mean(scores)
            median_score = statistics.median(scores)
            std_dev_score = statistics.stdev(scores)

            logger.info(f'\ttrial {trial.number}\'s results')

            logger.info(f'\t\tbest score: {best_score}')
            logger.info(f'\t\taverage score: {average_score}')
            logger.info(f'\t\tmedian score: {median_score}')
            logger.info(f'\t\tstd_dev score: {std_dev_score}')

            # optuna minimizes the objective by default, so we need to flip the sign to maximize
            cost = -1 * average_score

            return cost

        except (AssertionError, ValueError) as e:
            logger.error(f'pruning optimizer trial {trial} due to exception {e}')
            tb = traceback.format_exc()
            logger.error(tb)

            raise optuna.exceptions.TrialPruned()

    return objective
