#!/usr/bin/env python3
"""
Usage example:
python train_resnet.py -b 64 -lr 0.0001 --dataset-dir ./data --dump-dir ./saved_models/gradR --sparsity 0.9 -penalty 0.001 -T 8 -N 2048 -gpu 1 -m grad
"""

# NOTE: GPU selection ENV var must be set before importing torch for it to take effect.
import os
# Hardcode GPU 1 as requested (change to e.g. '0' or use args parsing if you want dynamic)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import torchvision
from torchvision import transforms

# spikingjelly imports
from spikingjelly.activation_based import functional as act_functional
# monitor API varies across versions; import and use safe wrapper below
try:
    from spikingjelly.activation_based import monitor as act_monitor
except Exception:
    act_monitor = None

import numpy as np
import sys
import time
import argparse
import tempfile
import shutil

# local model file for ResNet19 (must exist)
from model_resnet import ResNet19Net

sys.path.append('..')

from gradrewire import GradRewiring
from deeprewire import DeepRewiring

############## Reproducibility ##############
_seed_ = 2020
np.random.seed(_seed_)
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#############################################

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-penalty', type=float, default=1e-3)
parser.add_argument('-s', '--sparsity', type=float)
parser.add_argument('-gpu', type=str, default='1', help='which cuda visible device index (string)')
parser.add_argument('--dataset-dir', type=str, default='./data')
parser.add_argument('--dump-dir', type=str, default='./saved_models/gradR')
parser.add_argument('-T', type=int, default=8)
parser.add_argument('-N', '--epoch', type=int, default=2048)
parser.add_argument('-soft', action='store_true')
parser.add_argument('-test', action='store_true')
parser.add_argument(
    '-m', '--mode', choices=['deep', 'grad', 'no_prune'], default='no_prune')

# Epoch interval when recording data (firing rate, acc. on test set, etc.) on TEST set
parser.add_argument('-i1', '--interval-test', type=int, default=128)

# Step interval when recording data (loss, acc. on train set) on TRAIN set
parser.add_argument('-i2', '--interval-train', type=int, default=1024)
args = parser.parse_args()

# set CUDA_VISIBLE_DEVICES if user provided -gpu (keep above earlier hardcode precedence)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

batch_size = args.batch_size
learning_rate = args.learning_rate
dataset_dir = args.dataset_dir
dump_dir = args.dump_dir
T = args.T
penalty = args.penalty
s = args.sparsity
soft = args.soft
test = args.test
no_prune = (args.mode == 'no_prune')
i1 = args.interval_test
i2 = args.interval_train
N = args.epoch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Utility: atomic save & checkpoint functions
###############################################################################
def _atomic_save(obj, path):
    """
    Save `obj` to `path` atomically:
      1. write to a secure temp file in the system temp dir,
      2. fsync implicitly via torch.save, then move the file into `path`.
    This avoids the PyTorch 'invalid file name' error that can happen when
    saving directly to certain tmp-files in the target dir.
    """
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)

    # create a temp file in the system temp dir (not inside `d`)
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmpf.name
    tmpf.close()

    try:
        # Save to tmp file first
        torch.save(obj, tmp_path)

        # Move (rename) into final location. shutil.move handles cross-filesystem.
        shutil.move(tmp_path, path)
        print(f"[SAVE] Wrote file: {path}")

    except Exception as e:
        # cleanup temp if anything failed
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"[SAVE ERROR] Failed to save {path}: {e}")
        raise

def save_checkpoint(model, model_dir, optimizer_w=None, optimizer_bn=None, optimizer_all=None):
    """
    Atomic save: checkpoint.pth (dict) + net_weights.pth (state_dict).
    """
    os.makedirs(model_dir, exist_ok=True)
    ckpt = {
        'state_dict': model.state_dict(),
        'train_times': getattr(model, 'train_times', 0),
        'epochs': getattr(model, 'epochs', 0),
        'max_test_accuracy': getattr(model, 'max_test_acccuracy', getattr(model, 'max_test_accuracy', 0))
    }
    ckpt_path = os.path.join(model_dir, 'checkpoint.pth')
    weights_path = os.path.join(model_dir, 'net_weights.pth')
    _atomic_save(ckpt, ckpt_path)
    _atomic_save(ckpt['state_dict'], weights_path)

    # optimizers
    if optimizer_all is not None:
        _atomic_save(optimizer_all.state_dict(), os.path.join(model_dir, 'optim_all.pth'))
    if optimizer_w is not None:
        _atomic_save(optimizer_w.state_dict(), os.path.join(model_dir, 'optim_w.pth'))
    if optimizer_bn is not None:
        _atomic_save(optimizer_bn.state_dict(), os.path.join(model_dir, 'optim_bn.pth'))

def load_checkpoint_if_exists(model_class, model_dir, device='cuda', T=8):
    """
    Load checkpoint.pth or net_weights.pth or old net.pkl (back-compat).
    Returns (model, optim_states_dicts: dict)
    """
    ckpt_path = os.path.join(model_dir, 'checkpoint.pth')
    weights_path = os.path.join(model_dir, 'net_weights.pth')
    pkl_path = os.path.join(model_dir, 'net.pkl')  # legacy

    model = model_class(T=T).to(device)
    optim_states = {}

    if os.path.exists(ckpt_path):
        print(f'Loading checkpoint from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state)
        model.train_times = ckpt.get('train_times', getattr(model, 'train_times', 0))
        model.epochs = ckpt.get('epochs', getattr(model, 'epochs', 0))
        model.max_test_acccuracy = ckpt.get('max_test_accuracy', ckpt.get('max_test_acccuracy', 0))
        # try to load optimizer states (optional)
        for name in ['optim_all.pth', 'optim_w.pth', 'optim_bn.pth', 'optim_all', 'optim_w', 'optim_bn']:
            p = os.path.join(model_dir, name) if name.endswith('.pth') else None
        return model, optim_states

    if os.path.exists(weights_path):
        print(f'Loading state_dict from {weights_path}')
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        model.train_times = getattr(model, 'train_times', 0)
        model.epochs = getattr(model, 'epochs', 0)
        return model, optim_states

    if os.path.exists(pkl_path):
        print(f'Loading legacy pickled model from {pkl_path}')
        full = torch.load(pkl_path, map_location=device)
        if isinstance(full, dict) and 'state_dict' in full:
            model.load_state_dict(full['state_dict'])
            model.train_times = full.get('train_times', 0)
            model.epochs = full.get('epochs', 0)
            return model, {}
        try:
            if isinstance(full, torch.nn.Module):
                sd = full.state_dict()
                model.load_state_dict(sd)
                model.train_times = getattr(full, 'train_times', 0)
                model.epochs = getattr(full, 'epochs', 0)
                return model, {}
        except Exception:
            pass
    return model, {}

###############################################################################
# safe monitor wrapper (some spikingjelly versions don't have set_monitor)
###############################################################################
def safe_set_monitor(net, flag: bool):
    """
    If spikingjelly provides set_monitor in monitor module, call it.
    Otherwise, try toggling monitor attributes if present. If unavailable, no-op.
    """
    try:
        if act_monitor is not None and hasattr(act_monitor, 'set_monitor'):
            act_monitor.set_monitor(net, flag)
            return
    except Exception:
        pass
    # fallback: try per-module attribute 'monitor' existence; do nothing if not present
    # (we do not raise — it's optional instrumentation)
    return

###############################################################################
# Data loaders and main
###############################################################################
if __name__ == "__main__":

    file_prefix = 'lr-' + np.format_float_scientific(learning_rate, exp_digits=1, trim='-') + f'-b-{batch_size}-T-{T}'
    if not no_prune:
        file_prefix += '-penalty-' + np.format_float_scientific(penalty, exp_digits=1, trim='-')
    if soft:
        file_prefix = 'soft-' + file_prefix
    if s is not None and not no_prune:
        file_prefix += f'-s-{s}'
    file_prefix += '-' + args.mode

    log_dir = os.path.join(dump_dir, 'logs', file_prefix)
    model_dir = os.path.join(dump_dir, 'models', file_prefix)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Debug prints to help locate saved files
    print("Model will be saved to:", os.path.abspath(model_dir))
    print("Dump dir absolute:", os.path.abspath(dump_dir))
    print("CWD:", os.getcwd())
    print("Existing files in model_dir (before training):", os.listdir(model_dir) if os.path.exists(model_dir) else "NOT EXIST")

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=transform_train,
        download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=transform_test,
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True)

    # Load existing model or create a new one
    net, optim_states = load_checkpoint_if_exists(ResNet19Net, model_dir, device=device, T=T)

    # Ensure attributes exist
    if not hasattr(net, 'train_times'):
        net.train_times = 0
    if not hasattr(net, 'epochs'):
        net.epochs = 0
    if not hasattr(net, 'max_test_acccuracy'):
        net.max_test_acccuracy = 0

    print(f'Loaded model: epochs={net.epochs}, train_times={net.train_times}, max_test_acc={getattr(net,"max_test_acccuracy",0)}')

    # Use different optimizers for BN and other layers
    bn_params = []
    weight_params = []

    ttl_cnt = 0.0  # Number of all parameters
    w_cnt = 0.0  # Number of parameters to be pruned (BN excluded)

    # adjust these BN_list names to your net if needed
    BN_list = ['static_conv.1', 'conv.2', 'conv.5', 'conv.9', 'conv.12', 'conv.15']
    for name, param in net.named_parameters():
        if any(BN_name in name for BN_name in BN_list):
            bn_params += [param]
            ttl_cnt += param.numel()
        else:
            weight_params += [param]
            w_cnt += param.numel()
            ttl_cnt += param.numel()

    ###### TEST MODE ######
    if test:
        with torch.no_grad():
            # Try to enable monitor (safe)
            safe_set_monitor(net, True)

            spike_times = dict()
            for name, module in net.named_modules():
                if hasattr(module, 'monitor'):
                    spike_times[name] = 0

            test_sum = 0
            correct_sum = 0

            for img, label in test_data_loader:
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                out_spikes_counter = net(img)

                correct_sum += (out_spikes_counter.argmax(dim=1) == label).float().sum().item()
                test_sum += label.numel()

                for name, module in net.named_modules():
                    if hasattr(module, 'monitor'):
                        # monitor['s'] is a list, each element is of shape [batch_size, ...]
                        spike_times[name] += torch.sum(torch.from_numpy(np.concatenate(module.monitor['s'], axis=0)).cuda(), dim=0)

                act_functional.reset_net(net)

            test_accuracy = correct_sum / test_sum

            print('Firing Rates:')
            for k, v in spike_times.items():
                rate = (v / (T * len(test_dataset))).flatten().cpu().numpy()
                filename = 'rate-' + k + (('-no_prune.npy' if no_prune else f'-{np.format_float_scientific(penalty, exp_digits=1, trim="-")}.npy'))
                with open(os.path.join(log_dir, filename), 'wb') as f:
                    np.save(f, rate)

            if no_prune:
                print(f'Test Acc: {test_accuracy * 100:.3f}%')
            else:
                print('Sparsity:')
                zero_cnt = 0.0
                for name, param in net.named_parameters():
                    if not any(BN_name in name for BN_name in BN_list):
                        curr_zero_cnt = (param == 0.0).float().sum()
                        zero_cnt += curr_zero_cnt
                        print(f'{name}: {curr_zero_cnt / param.numel() * 100:.3f}%')

                sparsity_all = zero_cnt / ttl_cnt
                sparsity_w = zero_cnt / w_cnt

                print(f'Test Acc: {test_accuracy * 100:.3f}%, Sparsity (w/ BN): {sparsity_all * 100:.3f}%, Sparsity (w/o BN): {sparsity_w * 100:.3f}%')

    ###### TRAIN MODE ######
    else:
        # Build optimizers
        if no_prune:
            optimizer_all = Adam(net.parameters(), lr=learning_rate)
            if 'optim_all' in optim_states:
                try:
                    optimizer_all.load_state_dict(optim_states['optim_all'])
                    print("Loaded optimizer_all state.")
                except Exception as e:
                    print("Failed to load optimizer_all state:", e)
        else:
            if args.mode == 'grad':
                optimizer_w = GradRewiring(weight_params, lr=learning_rate, alpha=penalty, s=s)
            elif args.mode == 'deep':
                optimizer_w = DeepRewiring(weight_params, lr=learning_rate, l1=penalty, max_s=s, soft=soft)
            else:
                optimizer_w = None
            optimizer_bn = Adam(bn_params, lr=learning_rate)

            if 'optim_w' in optim_states and optimizer_w is not None:
                try:
                    optimizer_w.load_state_dict(optim_states['optim_w'])
                    print("Loaded optimizer_w state.")
                except Exception as e:
                    print("Failed to load optimizer_w state:", e)
            if 'optim_bn' in optim_states:
                try:
                    optimizer_bn.load_state_dict(optim_states['optim_bn'])
                    print("Loaded optimizer_bn state.")
                except Exception as e:
                    print("Failed to load optimizer_bn state:", e)

        writer_test = SummaryWriter(log_dir, flush_secs=600, purge_step=getattr(net, 'epochs', 0))
        writer_train = SummaryWriter(log_dir, flush_secs=600, purge_step=getattr(net, 'train_times', 0))

        print(net)

        max_test_accuracy = 0

        before_link = dict()
        after_link = dict()

        # Record initial connectivity
        if not no_prune:
            for name, param in net.named_parameters():
                if not any(BN_name in name for BN_name in BN_list):
                    before_link[name] = (param.abs() >= 1e-10)

        # Training Loop
        while True:
            net.train()
            print(f'Epoch {net.epochs}, {file_prefix}')

            time_start = time.time()
            for img, label in train_data_loader:
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                if no_prune:
                    optimizer_all.zero_grad()
                else:
                    if optimizer_w is not None:
                        optimizer_w.zero_grad()
                    optimizer_bn.zero_grad()

                out_spikes_counter = net(img)
                out_spikes_counter_frequency = out_spikes_counter / T

                loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, 10).float())
                loss.backward()

                if no_prune:
                    optimizer_all.step()
                else:
                    if optimizer_w is not None:
                        optimizer_w.step()
                    optimizer_bn.step()

                act_functional.reset_net(net)

                if net.train_times % i2 == 0:
                    correct_rate = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()
                    writer_train.add_scalar('train/acc', correct_rate, net.train_times)
                    writer_train.add_scalar('train/loss', loss.item(), net.train_times)

                net.train_times += 1

            # Evaluate at the end of training epoch
            net.eval()

            with torch.no_grad():
                if net.epochs % i1 == 0:
                    safe_set_monitor(net, True)
                    spike_times = dict()
                    for name, module in net.named_modules():
                        if hasattr(module, 'monitor'):
                            spike_times[name] = 0

                test_sum = 0
                correct_sum = 0
                for img, label in test_data_loader:
                    img = img.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)

                    out_spikes_counter = net(img)

                    correct_sum += (out_spikes_counter.argmax(dim=1) == label).float().sum().item()
                    test_sum += label.numel()

                    if net.epochs % i1 == 0:
                        for name, module in net.named_modules():
                            if hasattr(module, 'monitor'):
                                spike_times[name] += torch.sum(torch.from_numpy(np.concatenate(module.monitor['s'], axis=0)).cuda(), dim=0)

                    act_functional.reset_net(net)

                if net.epochs % i1 == 0:
                    for k, v in spike_times.items():
                        nonfire_prop = (v == 0).sum().cpu().item() / v.numel()
                        avg_firing_rate = v.mean() / (test_sum * T)
                        writer_test.add_scalar('nonfire_prop/' + k, nonfire_prop, net.epochs)
                        writer_test.add_scalar('avg_firing_rate/' + k, avg_firing_rate, net.epochs)

                test_accuracy = correct_sum / test_sum
                writer_test.add_scalar('test_acc', test_accuracy, net.epochs)

                if no_prune:
                    print(f'Test Acc: {test_accuracy * 100:.3f}%, Max Test Acc: {max_test_accuracy * 100:.3f}%')
                    if test_accuracy > max_test_accuracy:
                        max_test_accuracy = test_accuracy
                        net.max_test_acccuracy = max_test_accuracy
                    torch.save(optimizer_all.state_dict(), os.path.join(model_dir, 'optim_all.pth'))
                else:
                    zero_cnt = 0.0
                    for name, param in net.named_parameters():
                        if not any(BN_name in name for BN_name in BN_list):
                            curr_zero_cnt = (param.abs() < 1e-10).float().sum()
                            zero_cnt += curr_zero_cnt
                            writer_test.add_scalar('layer sparsity/' + name, curr_zero_cnt / param.numel(), net.epochs)

                            after_link[name] = (param.abs() >= 1e-10)
                            regrow_cnt = torch.logical_and(torch.logical_not(before_link[name]), after_link[name]).sum().item()
                            prune_cnt = torch.logical_and(torch.logical_not(after_link[name]), before_link[name]).sum().item()

                            writer_test.add_scalar('regrow_cnt/' + name, regrow_cnt, net.epochs)
                            writer_test.add_scalar('prune_cnt/' + name, prune_cnt, net.epochs)
                            writer_test.add_scalar('prune-regrow/' + name, prune_cnt - regrow_cnt, net.epochs)

                            before_link[name] = after_link[name].clone()

                    sparsity_all = zero_cnt / ttl_cnt
                    sparsity_w = zero_cnt / w_cnt

                    writer_test.add_scalar('sparsity/with bn', sparsity_all, net.epochs)
                    writer_test.add_scalar('sparsity/without bn', sparsity_w, net.epochs)

                    print(f'Test Acc: {test_accuracy * 100:.3f}%, Sparsity (w/ BN): {sparsity_all * 100:.3f}%, Sparsity (w/o BN): {sparsity_w * 100:.3f}%')
                    if optimizer_w is not None:
                        torch.save(optimizer_w.state_dict(), os.path.join(model_dir, 'optim_w.pth'))
                    torch.save(optimizer_bn.state_dict(), os.path.join(model_dir, 'optim_bn.pth'))

                # Save checkpoint (state_dict + metadata) atomically
                save_checkpoint(net, model_dir,
                                optimizer_w=(None if no_prune else (optimizer_w if 'optimizer_w' in locals() else None)),
                                optimizer_bn=(None if no_prune else optimizer_bn if 'optimizer_bn' in locals() else None),
                                optimizer_all=(None if not no_prune else optimizer_all if 'optimizer_all' in locals() else None))

                # also save a standalone state_dict snapshot every i1 epochs
                if net.epochs % i1 == 0:
                    snapshot_path = os.path.join(model_dir, f'net-{net.epochs}.pth')
                    _atomic_save(net.state_dict(), snapshot_path)

                # disable monitor if enabled
                if net.epochs % i1 == 0:
                    safe_set_monitor(net, False)

            net.epochs += 1

            time_end = time.time()
            print(f'Elapse: {time_end - time_start:.2f}s')

            if net.epochs > N:
                break

        # final save after training
        save_checkpoint(net, model_dir,
                        optimizer_w=(None if no_prune else (optimizer_w if 'optimizer_w' in locals() else None)),
                        optimizer_bn=(None if no_prune else optimizer_bn if 'optimizer_bn' in locals() else None),
                        optimizer_all=(None if not no_prune else optimizer_all if 'optimizer_all' in locals() else None))

        print("Training finished. Final checkpoint saved.")
