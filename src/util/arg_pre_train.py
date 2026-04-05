import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--steps', default=150000, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_steps_freq', default=10000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_flow_mamba', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.90, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=25, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/root/Vim/dataset/YaTC_datasets/USTC-TFC2016_MFR', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output/pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

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

    # pretrain tasks
    parser.add_argument('--pop', action='store_true', help='packet order prediction')
    parser.add_argument('--pop_loss_weight', default=0.01, type=float, 
                        help='packet order prediction loss weight')
    parser.add_argument("--num_packet", default=5, type=int,)
    parser.add_argument("--num_packet_byte", default=320, type=int,)
    parser.add_argument("--header_len", default=80, type=int,)
    parser.add_argument("--payload_len", default=240, type=int,)
    parser.add_argument("--stride_size", default=4, type=int,)
    parser.add_argument("--seq_len", default=50, type=int,
                        help="length of sequence for seq dataset")
    parser.add_argument("--dataset_type", default="byte", type=str,
                        help="type of dataset to use, either 'byte' or 'seq'")
    parser.add_argument("--seq_key", default="sizes", type=str,
                        choices=["sizes", "intervals"],
                        help="key to use for sequence data in seq dataset, either 'sizes' or 'intervals'")
    parser.add_argument("--size_key", default="sizes", type=str,
                        choices=["sizes", "signed_sizes"],
                        help="key to use for size data, either 'sizes' or 'signed_sizes'")
    parser.add_argument("--average", default="weighted", type=str,
                        choices=["weighted", "macro", "micro"],
                        help="average method for metrics")
    parser.add_argument('--byte_mask_ratio', default=0.90, type=float,
                        help='Masking ratio for byte modality (percentage of removed bytes).')
    parser.add_argument('--size_mask_ratio', default=0.15, type=float,
                        help='Masking ratio for size modality (percentage of removed sizes).')
    parser.add_argument('--iat_mask_ratio', default=0.15, type=float,
                        help='Masking ratio for interval modality (percentage of removed iats).')
    return parser