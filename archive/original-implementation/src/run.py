import sys; sys.path.append('.')
import argparse

from firelab.utils.training_utils import fix_random_seed, run_tensorboard
from firelab.config import Config

from src.trainers import GANTrainer


def read_cli_args():
    parser = argparse.ArgumentParser(description='Run meta-generation project')
    parser.add_argument('-c', '--config', required=True, type=str, help='Which config to run?')
    parser.add_argument('-tb', '--tb_port', type=int, help='Should start tensorboard?')

    args, _ = parser.parse_known_args()

    return args


def run(config_path: str, tb_port: int=None):
    config = Config.load(config_path)
    config = config.overwrite(Config.read_from_cli())

    if config.get('distributed_training.enabled'):
        import horovod.torch as hvd; hvd.init()
        fix_random_seed(config.random_seed + hvd.rank())
    else:
        fix_random_seed(config.random_seed)

    trainer = GANTrainer(config)

    if not tb_port is None and trainer.is_main_process():
        trainer.logger.info(f'Starting tensorboard on port {tb_port}')
        run_tensorboard(trainer.paths.experiment_dir, tb_port)

    trainer.start()


def main():
    args = read_cli_args()
    run(args.config, args.tb_port)


if __name__ == "__main__":
    main()
