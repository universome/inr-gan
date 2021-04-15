#!/usr/bin/env python

"""
Runs a __reproducible__ experiment on the current resources.
It copies the current codebase into a new project dir and runs an experiment from there.
"""
import os
import argparse

from utils import create_project_dir, get_git_hash_suffix, BASE_PROJECT_DIR


def main():
    args = read_args()

    run_experiment(args.experiments_dir, args.experiment_name, args.command, args.print)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Running the experiment')
    parser.add_argument('-d', '--experiments_dir', default='experiments', type=str,
                        help='A root directory where all the experiments are stored.')
    parser.add_argument('-e', '--experiment_name', type=str, help='Experiment name (for checkpoints saving and logging).')
    parser.add_argument('-c', '--command', type=str, help='Training command to execute')
    parser.add_argument('--print', action='store_true', help='If we should only print the final run command?')
    args, _ = parser.parse_known_args()

    return args


def run_experiment(experiments_dir: os.PathLike, experiment_name: str, command: str, print_only: bool):
    project_dir = os.path.join(BASE_PROJECT_DIR, experiments_dir, f'{experiment_name}{get_git_hash_suffix()}')
    hydra_run_dir_arg = f'hydra.run.dir={project_dir}/result'
    command = f'{command} {hydra_run_dir_arg}'

    is_project_created = create_project_dir(project_dir)
    assert is_project_created

    os.chdir(project_dir)

    # Just in case, let's save the running command?
    with open(os.path.join(project_dir, 'run_command.sh'), 'w') as f:
        f.write(command + '\n')

    if print_only:
        print(command)
    else:
        os.system(command)


if __name__ == "__main__":
    main()
