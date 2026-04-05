import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard.writer import SummaryWriter

import timm.optim.optim_factory as optim_factory

# import util.misc as misc
from util.misc import (
    count_parameters, init_distributed_mode, get_rank, get_world_size, 
    is_main_process
)
from util.arg_pre_train import get_args_parser
from util.loader_data import get_data_loader
from util.loader_model import get_model_mae
from engine_mm import pretrain_one_epoch
from contextlib import suppress


def main(args):
    init_distributed_mode(args)
    print(f"args.distributed: {args.distributed}")

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = get_world_size()
    global_rank = get_rank()
    data_path = os.path.join(args.data_path, "data-train.json") if os.path.exists(os.path.join(args.data_path, "data-train.json")) \
        else os.path.join(args.data_path, "data.json")
    data_loader_train, _ = get_data_loader(args, data_path, random_sampler=True)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the model
    model = get_model_mae(args)
    model.to(device)
    print("Model = %s" % str(model))
    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # amp about
    amp_autocast = suppress
    loss_scaler = "none"

    epochs = int(args.steps / len(data_loader_train)) + 1
    args.epochs = epochs

    print(f"Start training for {args.steps} steps")
    start_time = time.time()
    for epoch in range(0, epochs):
        # data_loader_train.sampler.set_epoch(epoch)
        train_stats = pretrain_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            args, log_writer=log_writer,
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }
        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    with open(os.path.join(args.output_dir, "train_stats.json"), mode="w", encoding="utf-8") as f:
        json.dump({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "total_time": total_time,
        }, f, indent=2)

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
