import itertools
import os
import subprocess
from pathlib import Path
from time import time
from typing import Iterable
from typing import Iterator

PYTHON_EXEC = 'venv/Scripts/python'
SCRIPT = 'src/examples/multi_arm_bandits.py'

SOURCE_DIRS = [
    'src',
]

EXPERIMENT_UID = str(int(time()))
OUTPUT_DIR = Path(os.path.join('local/experiments', EXPERIMENT_UID))


class TrialConfigurator:
    name: str

    def __iter__(self) -> Iterator[list[str]]:
        pass


class SameStatsVariableMachinesTrialConfigurator(TrialConfigurator):
    """
        variable parameters: n_machines = {2, 4, ..., 2 ** max_exponent}
        fixed parameters: max(p_win) = 1.0, min(p_win) = 0.0, and expected_value(p_win) = 0.5
    """

    def __init__(self, base_run_args: list[str], max_exponent: int = 5):
        self.base_run_args = base_run_args
        self.max_exponent = max_exponent

        self.name = 'SameStatsVariableMachines'

        self._n_machines = 1

    def __iter__(self) -> Iterator[list[str]]:
        self._n_machines = 1
        return self

    def __next__(self) -> list[str]:
        self._n_machines *= 2

        if self._n_machines >= 2 ** self.max_exponent:
            raise StopIteration

        # always have at least these 2 machines
        p_wins = [0.0, 1.0]

        # fill remaining machines, keeping max(p_win) = 1.0, min(p_win) = 0.0, and expected_value(p_win) = 0.5
        if self._n_machines > 2:
            p_win_increment = 1.0 / (self._n_machines - 2)
            p_wins.extend([p_win_increment / 2.0 + m * p_win_increment for m in range(self._n_machines - 2)])
            p_wins = sorted(p_wins)

        return self._generate_args(p_wins)

    def _generate_args(self, p_wins: Iterable[float]) -> list[str]:
        machine_args = itertools.chain.from_iterable([['-M', str(p_win)] for p_win in p_wins])
        return [*self.base_run_args, *machine_args]


def config_env() -> None:
    # setup environment variables
    os.environ['PYTHONPATH'] = os.pathsep.join([os.environ['PYTHONPATH'], *SOURCE_DIRS])

    # create output subdirectory
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()


def execute_run(run_id: int, run_args: list[str], output_path: Path) -> None:
    # 'python src/examples/multi_arm_bandits.py -M 1.0 -M 0.75 -M 0.25 -M 0.0 --steps 25000'
    with output_path.open('a') as file:
        file.write(f'\n***** beginning run {run_id} *****\n')
        file.flush()
        _ = subprocess.run([PYTHON_EXEC, SCRIPT, *run_args], text=True, check=True, stdout=file)
        file.write(f'\n***** ending run {run_id} *****\n')


def execute_trial(trial_configurator: TrialConfigurator, runs_per_trial: int) -> None:
    output_path = OUTPUT_DIR / f'{trial_configurator.name}.txt'
    with output_path.open('a') as file:
        file.write(f'\n***** beginning trial {trial_configurator.name} *****\n')

    for config_id, run_args in enumerate(trial_configurator):
        with output_path.open('a') as file:
            file.write(f'\n***** trial configuration {config_id}: {run_args}')

        for run_id in range(runs_per_trial):
            execute_run(run_id, run_args, output_path)


def execute_experiment(steps_per_run: int = 100, runs_per_trial: int = 2) -> None:
    config_env()

    base_run_args = [
        '--steps', str(steps_per_run)
    ]

    trials = [
        SameStatsVariableMachinesTrialConfigurator(base_run_args, max_exponent=5)
    ]

    for trial in trials:
        execute_trial(trial, runs_per_trial)


if __name__ == '__main__':
    execute_experiment()
