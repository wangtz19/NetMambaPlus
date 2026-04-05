import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_steps_freq', default=5000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='flow_mamba_mid_cls', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
#20
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--finetune', default='././output/pretrain/checkpoint.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--data_path', default='./ddos_datasets/flow_image', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output/finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--ckpt_dir', default='./output/finetune',
                        help='path where to load checkpoints')
    parser.add_argument('--log_dir', default='./output/finetune',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)
    # parser.add_argument("--arr_length", default=1600, type=int,)
    parser.add_argument("--num_packet", default=5, type=int,)
    parser.add_argument("--num_packet_byte", default=320, type=int,)
    parser.add_argument("--stride_size", default=4, type=int,)
    parser.add_argument("--class_balance", action="store_true",
                        help="whether to use class-balanced loss")
    parser.add_argument("--class_balance_beta", default=0.999, type=float,)
    parser.add_argument("--ldam", action="store_true",
                        help="whether to use LDAM loss")
    parser.add_argument("--data_ratio", default=1.0, type=float,)
    # dataset type
    parser.add_argument("--dataset_type", default="byte", type=str,
                        help="type of dataset to use, either 'byte' or 'seq'")
    parser.add_argument("--seq_len", default=50, type=int,
                        help="length of sequence for seq dataset")
    parser.add_argument("--seq_key", default="sizes", type=str,
                        choices=["sizes", "intervals"],
                        help="key to use for sequence data in seq dataset, either 'sizes' or 'intervals'")
    parser.add_argument("--size_key", default="sizes", type=str,
                        choices=["sizes", "signed_sizes"],
                        help="key to use for size data, either 'sizes' or 'signed_sizes'")
    # sanity check for loading pre-trained model
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="whether to train the model from scratch without loading pre-trained weights")
    parser.add_argument("--average", default="weighted", type=str,
                        choices=["weighted", "macro", "micro"],
                        help="average method for metrics")

    return parser