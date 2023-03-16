from accelerate import Accelerator
from omegaconf import open_dict
import hydra
import torch
import time

from .utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
)


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    accelerator = Accelerator(cpu=args.device == "cpu")
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    model = get_model(args, config)
    tokenizer = get_tokenizer(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    logger.log_args(args)

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    if args.model.compile:
        model = torch.compile(model)

    with open_dict(args):
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval(model, test_dataloader, logger, args, tokenizer)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger,
                    args, tokenizer)
    else:
        train(model, train_dataloader, test_dataloader, accelerator,
              lr_scheduler, optimizer, logger, args, tokenizer)

    logger.finish()


if __name__ == "__main__":
    main()
