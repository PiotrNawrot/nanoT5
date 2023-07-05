import torch
import os

from accelerate.utils import set_seed
from omegaconf import open_dict
from .logging_utils import Logger
from hydra.utils import to_absolute_path


def check_args_and_env(args):
    assert args.optim.batch_size % args.optim.grad_acc == 0

    # Train log must happen before eval log
    assert args.eval.every_steps % args.logging.every_steps == 0

    if args.device == 'gpu':
        assert torch.cuda.is_available(), 'We use GPU to train/eval the model'

    assert not (args.eval_only and args.predict_only)

    if args.predict_only:
        assert args.mode == 'ft'


def opti_flags(args):
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.precision == 'bf16' and args.device == 'gpu' and args.model.klass == 'local_t5':
        args.model.add_config.is_bf16 = True


def update_args_with_env_info(args):
    with open_dict(args):
        slurm_id = os.getenv('SLURM_JOB_ID')

        if slurm_id is not None:
            args.slurm_id = slurm_id
        else:
            args.slurm_id = 'none'

        args.working_dir = os.getcwd()


def update_paths(args):
    if args.mode == 'ft':
        args.data.exec_file_path = to_absolute_path(args.data.exec_file_path)
        args.data.data_dir = to_absolute_path(args.data.data_dir)
        args.data.task_dir = to_absolute_path(args.data.task_dir)


def setup_basics(accelerator, args):
    check_args_and_env(args)
    update_args_with_env_info(args)
    update_paths(args)
    opti_flags(args)

    if args.seed is not None:
        set_seed(args.seed)

    logger = Logger(args=args, accelerator=accelerator)

    return logger
