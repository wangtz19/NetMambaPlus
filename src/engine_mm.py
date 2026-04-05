import math
import sys
from typing import Iterable, Optional, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from tqdm import tqdm
import json
import os
import numpy as np
from torch.autograd import Variable, Function
import random
from typing import List


def get_data_processors():
    def process_byte_size(data, device):
        x_byte, x_size, targets = data
        x_byte = x_byte.to(device, non_blocking=True)
        x_size = x_size.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True) if targets is not None else None
        return {'x_byte': x_byte, 'x_size': x_size, 'targets': targets}
    
    def process_byte_size_interval(data, device):
        x_byte, x_size, x_interval, targets = data
        x_byte = x_byte.to(device, non_blocking=True)
        x_size = x_size.to(device, non_blocking=True)
        x_interval = x_interval.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True) if targets is not None else None
        return {'x_byte': x_byte, 'x_size': x_size, 'x_interval': x_interval, 'targets': targets}
    
    return {
        "byte_size": process_byte_size,
        "byte_size_interval": process_byte_size_interval,
    }


def get_model_forward_fn(byte_mask_ratio=0.90, size_mask_ratio=0.15, iat_mask_ratio=0.15):
    def forward_byte_size(model, inputs):
        return model(inputs['x_byte'], inputs['x_size'])
    
    def forward_byte_size_interval(model, inputs):
        return model(inputs['x_byte'], inputs['x_size'], inputs['x_interval'],
                     byte_mask_ratio=byte_mask_ratio, size_mask_ratio=size_mask_ratio,
                     iat_mask_ratio=iat_mask_ratio)

    return {
        "byte_size": forward_byte_size,
        "byte_size_interval": forward_byte_size_interval,
    }


def pretrain_one_epoch(model: torch.nn.Module,
                    data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast,
                    args, log_writer=None,):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('steps', misc.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    process_data = get_data_processors()[args.dataset_type]
    forward_model = get_model_forward_fn(
        byte_mask_ratio=args.byte_mask_ratio, size_mask_ratio=args.size_mask_ratio, 
        iat_mask_ratio=args.iat_mask_ratio
    )[args.dataset_type]

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    steps_of_one_epoch = len(data_loader)
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        steps = steps_of_one_epoch * epoch + data_iter_step
        metric_logger.update(steps=int(steps))
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        inputs = process_data(data, device)
        with amp_autocast():
            outputs = forward_model(model, inputs)
        
        if args.dataset_type == "byte_size":
            byte_loss, size_loss = outputs
            loss = byte_loss + size_loss
        elif args.dataset_type in ["byte_size_interval"]:
            byte_loss, size_loss, interval_loss = outputs
            loss = byte_loss + size_loss + interval_loss
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(byte_loss=byte_loss.item())
        metric_logger.update(size_loss=size_loss.item())
        if args.dataset_type in ["byte_size_interval"]:
            metric_logger.update(interval_loss=interval_loss.item()) # type: ignore
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train/byte_loss', byte_loss.item(), epoch_1000x)
            log_writer.add_scalar('train/size_loss', size_loss.item(), epoch_1000x)
            if args.dataset_type in ["byte_size_interval"]:
                log_writer.add_scalar('train/interval_loss', interval_loss.item(), epoch_1000x) # type: ignore
            log_writer.add_scalar('train/lr', lr, epoch_1000x)
        if args.output_dir and steps % args.save_steps_freq == 0 and epoch > 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "steps": steps,
            }, os.path.join(args.output_dir, f"checkpoint-step{steps}.pth"))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, 
                    args, max_norm: float = 0, log_writer=None,):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    process_data = get_data_processors()[args.dataset_type]
    forward_model = get_model_forward_fn()[args.dataset_type]
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        inputs = process_data(data, device)
        outputs = forward_model(model, inputs)
        logits = outputs["logits"]
        loss = criterion(logits, inputs["targets"])
        metric_logger.update(loss=loss.item())
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train/lr', max_lr, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader: DataLoader, model: torch.nn.Module, device, args,
             return_logits=False):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    pred_all = []
    target_all = []
    logits_all = []
    loss_all = []

    process_data = get_data_processors()[args.dataset_type]
    forward_model = get_model_forward_fn()[args.dataset_type]

    for batch in metric_logger.log_every(data_loader, 10, header):
        inputs = process_data(batch, device)
        with torch.cuda.amp.autocast():
            logits = forward_model(model, inputs)["logits"]
        targets = inputs["targets"]
        loss = criterion(logits, targets)
        loss_all.append(loss.item())
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        if return_logits:
            logits_all.extend(logits.cpu().tolist())
        pred_all.extend(pred[0].cpu())
        target_all.extend(targets.cpu())
        acc1, acc3 = accuracy(logits, targets, topk=(1, min(3, logits.shape[1])))
        batch_size = logits.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item()/100, n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item()/100, n=batch_size)

    macro = precision_recall_fscore_support(target_all, pred_all, average=args.average)
    cm = confusion_matrix(target_all, pred_all).tolist()

    # compute acc, precision, recall, f1 for each class
    acc = accuracy_score(target_all, pred_all)
    pre_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(target_all, pred_all, average=None)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.4f} Acc@3 {top3.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss))
    print(
        '* Pre {macro_pre:.4f} Rec {macro_rec:.4f} F1 {macro_f1:.4f}'
        .format(macro_pre=macro[0], macro_rec=macro[1],
                    macro_f1=macro[2]))

    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['weighted_pre'] = macro[0]
    test_state['weighted_rec'] = macro[1]
    test_state['weighted_f1'] = macro[2]
    test_state['cm'] = cm
    test_state['acc'] = acc
    test_state['pre_per_class'] = pre_per_class
    test_state['rec_per_class'] = rec_per_class
    test_state['f1_per_class'] = f1_per_class
    test_state['support_per_class'] = support_per_class
    test_state['loss'] = np.mean(loss_all)
    if return_logits:
        test_state['logits'] = logits_all
        test_state['labels'] = target_all

    return test_state


@torch.no_grad()
def evaluate_per_class(data_loader: DataLoader, model: torch.nn.Module, device, args):
    # switch to evaluation mode
    model.eval()
    process_data = get_data_processors()[args.dataset_type]
    forward_model = get_model_forward_fn()[args.dataset_type]
    num_classes = args.nb_classes
    total_preds = [[] for _ in range(num_classes)]
    total_labels = [[] for _ in range(num_classes)]
    for batch in tqdm(data_loader, desc="Evaluation"):
        inputs = process_data(batch, device)
        with torch.cuda.amp.autocast():
            logits = forward_model(model, inputs)["logits"]
        targets = inputs["targets"]
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        for i in range(len(targets)):
            label = targets[i].item()
            total_preds[label].append(pred[0][i].cpu().item())
            total_labels[label].append(label)
    metric_per_class = {}
    for c in range(num_classes):
        acc = accuracy_score(total_labels[c], total_preds[c])
        pre, rec, f1, _ = precision_recall_fscore_support(total_labels[c], total_preds[c], average='weighted', zero_division=0)
        metric_per_class[c] = {
            "acc": acc,
            "pre": pre,
            "rec": rec,
            "f1": f1,
            "num_samples": len(total_labels[c])
        }
    return metric_per_class


def get_cls_token_per_class(data_loader: DataLoader, model: torch.nn.Module, device, args):
    # switch to evaluation mode
    model.eval()
    process_data = get_data_processors()[args.dataset_type]
    forward_model = get_model_forward_fn()[args.dataset_type]
    num_classes = args.nb_classes
    total_features = [[] for _ in range(num_classes)]
    for batch in tqdm(data_loader, desc="Get features"):
        inputs = process_data(batch, device)
        with torch.cuda.amp.autocast():
            outputs = forward_model(model, inputs)
            features = outputs["cls_token"]
            preds = outputs["logits"].argmax(dim=1)
        if args.use_pred_label:
            targets = preds
        else:
            targets = inputs["targets"]
        for i in range(len(targets)):
            label = targets[i].item()
            total_features[label].append(features[i].cpu().tolist())
    return total_features


@torch.no_grad()
def compute_forward(data_loader, model, device, output_key="logits"):
    model.eval()
    assert output_key in ["logits", "cls_token", "confidence"]
    total_outputs, total_labels = [], []
    for batch in data_loader:
        x_byte, x_size, x_interval, targets = batch
        x_byte = x_byte.to(device, non_blocking=True)
        x_size = x_size.to(device, non_blocking=True)
        x_interval = x_interval.to(device, non_blocking=True)
        outputs = model(x_byte, x_size, x_interval)[output_key]
        total_outputs.extend(outputs.cpu().tolist())
        total_labels.extend(targets.cpu().tolist())
    return total_outputs, total_labels


import time
import gc

@torch.no_grad()
def evaluate_speed_test(data_loader: DataLoader, model: torch.nn.Module, device, args):
    # switch to evaluation mode
    model.eval()
    model_mem = torch.cuda.memory_allocated() / (1024**2)
    res_list = []
    process_data = get_data_processors()[args.dataset_type]
    forward_model = get_model_forward_fn()[args.dataset_type]
    for i in tqdm(range(3, 11), desc="Batch size"):
        # reset memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        batch_size = 2 ** i
        if i == 3:
            batch_size = 2 ** 10
        data_loader_tmp = torch.utils.data.DataLoader(data_loader.dataset, sampler=data_loader.sampler, # type: ignore
                            batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True, drop_last=False)
        pred_all = []
        start_time = time.time()
        for batch in data_loader_tmp:
            if time.time() - start_time > 30:
                break
            inputs = process_data(batch, device)
            with torch.cuda.amp.autocast():
                logits = forward_model(model, inputs)["logits"]
            _, pred = logits.topk(1, 1, True, True)
            pred = pred.t()
            pred_all.extend(pred[0].cpu())
            # target_all.extend(target.cpu())
        end_time = time.time()
        max_mem = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.empty_cache()
        gc.collect()

        if i == 3: # only for warming up
            continue
        res_list.append({
            "batch size": batch_size,
            "time (s)": end_time - start_time,
            "total smaples": len(pred_all),
            "speed (sample per second)": len(pred_all) / (end_time - start_time),
            "max memory consumption (MB)": max_mem,
            "model memory consumption (MB)": model_mem
        })
   
    with open(os.path.join(args.output_dir, "speed_test.json"), "w") as f:
        json.dump(res_list, f, indent=2)

    return None