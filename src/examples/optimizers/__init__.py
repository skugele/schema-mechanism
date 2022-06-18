import logging

from optuna import Trial

logger = logging.getLogger(__name__)


def display_hyper_parameters(trial: Trial) -> None:
    logger.info(f'{"-" * 80}')
    for param, value in trial.params.items():
        logger.info(f'\t{param}:{value}')
    logger.info(f'{"-" * 80}')
