import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard.writer import SummaryWriter
from timm.models.layers import trunc_normal_

import os
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import count_parameters
from util.arg_fine_tune import get_args_parser
from util.loader_data import get_data_loader, get_num_sample_per_cls
from util.loader_model import get_model_classifier
from util.metric_eval import draw_confusion_matrix
from util.loss import LDAMLoss

from engine_mm import train_one_epoch, evaluate
import random


def get_criterion(args, num_sample_per_cls, device):
    # prepare class balance weights
    effective_num = 1.0 - torch.pow(args.class_balance_beta, num_sample_per_cls)
    weights = (1.0 - args.class_balance_beta) / effective_num
    weights = weights / torch.sum(weights) * args.nb_classes
    weights = weights.to(device)

    if args.ldam:
        criterion = LDAMLoss(cls_num_list=num_sample_per_cls.tolist(), 
                            device=device,
                            weight=weights if args.class_balance else None,
                            label_smoothing=args.smoothing)
    elif args.class_balance:
        criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=args.smoothing)
    elif args.smoothing > 0.:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    return criterion


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ensure deterministic behavior (may slow down)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    data_loader_train, _ = get_data_loader(
        args, os.path.join(args.data_path, "data-train.json"),
        data_ratio=args.data_ratio, random_sampler=True)
    num_sample_per_cls = get_num_sample_per_cls(data_loader_train, args.nb_classes, args.dataset_type)
    data_loader_val, idx2label = get_data_loader(
        args, os.path.join(args.data_path, "data-valid.json"),)

    model = get_model_classifier(args)
    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if args.eval:
        best_checkpoint = torch.load(os.path.join(args.ckpt_dir, "checkpoint-best.pth"), map_location='cpu')
        msg = model.load_state_dict(best_checkpoint['model'], strict=False)
        print(msg)
        model.to(device)
        model.eval()
        test_result = evaluate(data_loader_test, model, device, args)
        print(test_result['cm'])
        if args.average == "weighted":
            average = ""
        else:
            average = f"_{args.average}"
        with open(os.path.join(args.output_dir, f"test_stats{average}.json"), mode="w", encoding="utf-8") as f:
            json.dump({
                "loss": test_result["loss"],
                "acc": test_result["acc"],
                "weighted_pre": test_result["weighted_pre"],
                "weighted_rec": test_result["weighted_rec"],
                "weighted_f1": test_result["weighted_f1"],
                "cm": test_result["cm"],
            }, f, indent=2)
        draw_confusion_matrix(test_result['cm'], idx2label,
                              save_path=os.path.join(args.output_dir, "confusion_matrix.pdf"))
        print(f"Accuracy of the network on test samples: {test_result['acc1']:.1f}%")
        exit(0)

    log_writer = SummaryWriter(log_dir=args.log_dir)
 
    if os.path.exists(args.finetune):
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        # interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
    elif args.train_from_scratch:
        print("No pre-trained model is loaded, training from scratch")
    else:
        raise ValueError("No pre-trained model is loaded, please specify --finetune or --train_from_scratch")

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
                                        no_weight_decay_list=model.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    criterion = get_criterion(args, num_sample_per_cls, device)
    print("criterion = %s" % str(criterion))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0
    best_acc = 0
    best_epoch = 0
    total_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        total_epoch += 1
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, None, None,
            args, max_norm=args.clip_grad, log_writer=log_writer,
        )
        test_stats = evaluate(data_loader_val, model, device, args)
        if test_stats["acc"] > best_acc:
            best_acc = test_stats["acc"]
            best_epoch = epoch
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, os.path.join(args.output_dir, "checkpoint-best.pth"))
        
        log_writer.add_scalar("val/acc", test_stats["acc"], epoch)
        log_writer.add_scalar("val/weighted_f1", test_stats["weighted_f1"], epoch)
        log_writer.add_scalar("val/loss", test_stats["loss"], epoch)

        print(f"Accuracy of the network on test samples: {test_stats['acc1']:.4f}")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"F1 of the network on test samples: {test_stats['weighted_f1']:.4f}")
        max_f1 = max(max_f1, test_stats["weighted_f1"])
        print(f'Max Accuracy: {max_accuracy:.4f}')
        print(f'Max F1: {max_f1:.4f}')

        test_stats.pop("cm") # do not print confusion matrix
        test_stats.pop("pre_per_class")
        test_stats.pop("rec_per_class")
        test_stats.pop("f1_per_class")
        test_stats.pop("support_per_class")

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v if not isinstance(v, np.ndarray) else v.tolist() for k, v in train_stats.items()},
                     **{f'valid_{k}': v if not isinstance(v, np.ndarray) else v.tolist() for k, v in test_stats.items()},
                     'n_parameters': n_parameters}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    with open(os.path.join(args.output_dir, "train_stats.json"), mode="w", encoding="utf-8") as f:
        json.dump({
            "batch_size": args.batch_size,
            "epochs": total_epoch,
            "total_time": total_time,
            "best_valid_acc": best_acc,
            "best_epoch": best_epoch,
        }, f, indent=2)
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    best_checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint-best.pth"), map_location='cpu')
    model.load_state_dict(best_checkpoint['model'])
    data_loader_test, _ = get_data_loader(
        args, os.path.join(args.data_path, "data-test.json"),)
    test_result = evaluate(data_loader_test, model, device, args)
    with open(os.path.join(args.output_dir, "test_stats.json"), mode="w", encoding="utf-8") as f:
        json.dump({
            "loss": test_result["loss"],
            "acc": test_result["acc"],
            # "acc1": test_result["acc1"],
            # "acc5": test_result["acc5"],
            "weighted_pre": test_result["weighted_pre"],
            "weighted_rec": test_result["weighted_rec"],
            "weighted_f1": test_result["weighted_f1"],
            "cm": test_result["cm"],
        }, f, indent=2)
    draw_confusion_matrix(test_result['cm'], idx2label,
                          save_path=os.path.join(args.output_dir, "confusion_matrix.pdf"))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)