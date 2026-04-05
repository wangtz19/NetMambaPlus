import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(parent_dir, "models")
sys.path.append(model_path)

from models import (models_net_trans, models_net_mamba,
                     models_net_trans_fuse3, models_net_mamba_fuse3)

    
def get_model_source(args):
    if args.model.startswith("net_mamba"):
        model_source = models_net_mamba
    elif args.model.startswith("fuse3_mamba"):
        model_source = models_net_mamba_fuse3
    elif args.model.startswith("net_bt") or args.model.startswith("net_bgt") \
        or args.model.startswith("net_ft") or args.model.startswith("net_fgt") \
        or args.model.startswith("net_lt"):
        model_source = models_net_trans
    elif args.model.startswith("fuse3_"):
        model_source = models_net_trans_fuse3
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model_source


def get_model_classifier(args, model_name=None):
    model_source = get_model_source(args)
    model_name = args.model if model_name is None else model_name
    model = model_source.__dict__[model_name](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            arr_length=args.num_packet * args.num_packet_byte,
            stride_size=args.stride_size,
            seq_len=args.seq_len,
            size_key=args.size_key,
        )
    return model


def get_model_mae(args):
    model_source = get_model_source(args)
    model = model_source.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            arr_length=args.num_packet * args.num_packet_byte,
            stride_size=args.stride_size,
            seq_len=args.seq_len,
            size_key=args.size_key,
        )
    return model