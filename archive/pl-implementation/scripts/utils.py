import os
import shutil
import subprocess
from distutils.dir_util import copy_tree
from shutil import copyfile
from typing import List, Optional

import click
import git

BASE_PROJECT_DIR = os.getcwd()
OBJECTS_TO_COPY = [os.path.join(BASE_PROJECT_DIR, d) for d in [
    'scripts',
    'configs',
    'dinr',
    'train_net.py',
    'requirements.txt',
    'setup.py',
]]

# A list of objects that are static and too big
# to be copy-pasted for each experiment
SYMLINKS_TO_CREATE = [os.path.join(BASE_PROJECT_DIR, s) for s in [
    'data',
    'inception_stats',
]]


def copy_objects(target_dir: os.PathLike, objects_to_copy: List[os.PathLike]):
    for src_path in objects_to_copy:
        trg_path = os.path.join(target_dir, os.path.basename(src_path))

        if os.path.islink(src_path):
            os.symlink(os.readlink(src_path), trg_path)
        elif os.path.isfile(src_path):
            copyfile(src_path, trg_path)
        elif os.path.isdir(src_path):
            copy_tree(src_path, trg_path)
        else:
            raise NotImplementedError(f"Unknown object type: {src_path}")


def create_symlinks(target_dir: os.PathLike, symlinks_to_create: List[os.PathLike]):
    """
    Creates symlinks to the given paths
    """
    for src_path in symlinks_to_create:
        trg_path = os.path.join(target_dir, os.path.basename(src_path))

        if os.path.islink(src_path):
            # Let's not create symlinks to symlinks
            # Since dropping the current symlink will break the experiment
            os.symlink(os.readlink(src_path), trg_path)
        else:
            print(f'Creating a symlink to {src_path}, so try not to delete it occasionally!')
            os.symlink(src_path, trg_path)


def is_git_repo(path=BASE_PROJECT_DIR):
    try:
        _ = git.Repo(path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


def create_project_dir(project_dir: os.PathLike) -> bool:
    if is_git_repo() and are_there_uncommitted_changes():
        if click.confirm("There are uncommited changes. Continue?", default=False):
            print('Ok...')
        else:
            print('Stopping.')
            return False

    if os.path.exists(project_dir):
        if click.confirm(f'Dir {project_dir} already exists. Remove it?', default=False):
            shutil.rmtree(project_dir)
        else:
            print('User refused to delete an existing project dir.')
            return False

    os.makedirs(project_dir)
    copy_objects(project_dir, OBJECTS_TO_COPY)
    create_symlinks(project_dir, SYMLINKS_TO_CREATE)

    if is_git_repo() and are_there_uncommitted_changes():
        print('Warning: there are uncommited changes!')

    print(f'Created a project dir: {project_dir}')

    return True


def get_git_hash() -> Optional[str]:
    if not is_git_repo():
        return None

    try:
        return subprocess \
            .check_output(['git', 'rev-parse', '--short', 'HEAD']) \
            .decode("utf-8") \
            .strip()
    except:
        return None


def get_git_hash_suffix() -> str:
    git_hash: Optional[str] = get_git_hash()
    git_hash_suffix = "" if git_hash is None else f"-{git_hash}"

    return git_hash_suffix


def are_there_uncommitted_changes() -> bool:
    return len(subprocess.check_output('git status -s'.split()).decode("utf-8")) > 0
