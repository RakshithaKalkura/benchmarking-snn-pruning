#!/usr/bin/env python3
"""
estimate_sata_energy.py

Usage (example):
python estimate_sata_energy.py \
  --checkpoint /path/to/checkpoint_max_acc1.pth \
  --model ResNet19SNN \
  --dataset CIFAR10 \
  --data-path ./data \
  --batch-size 128 \
  --T 8 \
  --out csv_results.csv

The script loads model via the same imports used by your main.py:
from models import cifar10, cifar10dvs, sew_resnet, resnet

It then runs the test set once to measure SOPs and computes energy estimates.

Defaults chosen to match your repo comments:
    E_SOP = 0.9 pJ (0.9e-12 J)
    E_MAC = 4.6 pJ (4.6e-12 J)
Training multiplier default: backward_mul = 2.0  (training cost ~= 3x inference)

Output: prints summary and writes a CSV row to --out
"""

import argparse
import os
import time
import csv
from math import ceil
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision import transforms

# Try to import repo models & utils as in your main.py
try:
    from models import cifar10, cifar10dvs, sew_resnet, resnet
except Exception as e:
    raise ImportError("Couldn't import models package from repo. Run this from repo root. Error: " + str(e))

# Try to import SOPMonitor from your utils (you used it earlier)
SOPMonitor = None
try:
    from utils import SOPMonitor
except Exception:
    SOPMonitor = None

# fallback spike counter (best-effort)
def _install_spike_hooks(model):
    """
    Best-effort: count positive spikes from modules that expose 'monitor' lists
    or for spikingjelly neuron modules.
    Returns hooks list and a dict spike_counts[name] = list of totals per batch
    """
    spike_counts = {}
    hooks = []

    # prefer modules that have attribute 'monitor' we saw used earlier
    for name, module in model.named_modules():
        if hasattr(module, 'monitor'):
            # monitor['s'] will be used during forward; we accumulate later by reading module.monitor
            spike_counts[name] = 0
    # If none found, attempt to hook LIFNode-like forward outputs (dangerous)
    # For now, we only return empty hooks; SOPMonitor path is preferred.
    return hooks, spike_counts

def reset_model_net(model):
    # Use your repo style reset; model may require spikingjelly.reset functions
    try:
        from spikingjelly.activation_based.functional import reset_net
        reset_net(model)
    except Exception:
        pass

def load_test_data(dataset_type, data_path, batch_size, workers, T):
    if dataset_type == 'CIFAR10':
        mean=(0.4914, 0.4822, 0.4465)
        std=(0.2470, 0.2435, 0.2616)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    elif dataset_type == 'CIFAR100':
        mean = [n/255. for n in [129.3,124.1,112.4]]
        std = [n/255. for n in [68.2,65.4,70.4]]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_test = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError("Unsupported dataset_type: " + str(dataset_type))
    loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return dataset_test, loader

def construct_model_by_name(model_name, T, num_classes):
    # mirror the logic in your main.py model loading
    # check cifar10
    if model_name in cifar10.__dict__:
        return cifar10.__dict__[model_name](T=T, num_classes=num_classes)
    elif model_name in cifar10dvs.__dict__ if 'cifar10dvs' in globals() else False:
        return cifar10dvs.__dict__[model_name]()  # usually not used for CIFAR10
    elif model_name in sew_resnet.__dict__:
        return sew_resnet.__dict__[model_name](T=T, num_classes=num_classes)
    elif model_name in resnet.__dict__:
        # your resnet module might expose ResNet19SNN or ResNet19 etc
        return resnet.__dict__[model_name](T=T, num_classes=num_classes)
    else:
        # fallback - try calling resnet.ResNet19 if user provided that name
        if hasattr(resnet, model_name):
            return getattr(resnet, model_name)(T=T, num_classes=num_classes)
        raise ValueError(f"Model {model_name} not found in models package. Choose one of: {list(cifar10.__dict__.keys()) + list(getattr(sew_resnet, '__dict__', {}).keys()) + list(resnet.__dict__.keys())}")

def measure_sops_using_sopmonitor(model, data_loader, device, args):
    # Use your repo's SOPMonitor if available
    mon = None
    if SOPMonitor is not None:
        mon = SOPMonitor(model)
        mon.enable()
    else:
        mon = None

    model.eval()
    total_samples = 0
    # We'll accumulate per-layer list of tensors (mon.monitored_layers -> lists)
    with torch.no_grad():
        for idx, (imgs, targets) in enumerate(data_loader):
            imgs = imgs.float().to(device)
            targets = targets.to(device)
            out = model(imgs)
            # reset net (your repo uses spikingjelly.functional.reset_net)
            reset_model_net(model)
            batch_size = imgs.shape[0]
            total_samples += batch_size

    # compute SOPs from mon (mon.monitored_layers contains lists of per-batch tensors)
    if SOPMonitor is not None:
        sops = 0.0
        for name in mon.monitored_layers:
            sublist = mon[name]
            sop = torch.cat(sublist).mean().item()  # mean spikes per sample for this layer
            sops += sop
        # sops returned in *spike counts per sample across model* — convert to SOPs as used in repo (they divided by 1e9 later).
        return sops, total_samples
    else:
        # no SOPMonitor; cannot compute exact SOPs — return None
        return None, total_samples

def measure_macs_with_thop(model, input_size, device):
    try:
        from thop import profile
    except Exception:
        return None
    model_cpu = model.to('cpu')
    dummy = torch.randn(*input_size)
    try:
        macs, params = profile(model_cpu, inputs=(dummy,), verbose=False)
        # macs is MAC count
        return macs
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='path to checkpoint (checkpoint.pth or state_dict)')
    ap.add_argument('--model', required=True, help='model class name (e.g. ResNet19SNN or ResNet19)')
    ap.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10','CIFAR100'])
    ap.add_argument('--data-path', default='./data')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--E_sop_pJ', type=float, default=0.9, help='pJ per SOP (default 0.9 pJ)')
    ap.add_argument('--E_mac_pJ', type=float, default=4.6, help='pJ per MAC (default 4.6 pJ)')
    ap.add_argument('--backward_mul', type=float, default=2.0, help='backward multiplier (default 2.0 => train ~3x inference)')
    ap.add_argument('--epochs', type=int, default=200, help='used for training energy estimate (if None, not computed)')
    ap.add_argument('--out', default='sata_energy_results.csv')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load model
    num_classes = 10 if args.dataset == 'CIFAR10' else 100
    print("Constructing model:", args.model, "T=", args.T)
    model = construct_model_by_name(args.model, T=args.T, num_classes=num_classes)
    model = model.to(device)

    # load checkpoint
    ck = torch.load(args.checkpoint, map_location='cpu')
    # Accept either { 'state_dict': ... } or plain state_dict
    if isinstance(ck, dict) and 'state_dict' in ck:
        sd = ck['state_dict']
    else:
        sd = ck
    try:
        model.load_state_dict(sd)
    except Exception as e:
        # try stripping 'module.' prefix (from DDP)
        new_sd = {}
        for k,v in sd.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[len('module.'):]
            new_sd[nk] = v
        model.load_state_dict(new_sd)
    model.eval()

    # prepare test loader
    dataset_test, test_loader = load_test_data(args.dataset, args.data_path, args.batch_size, args.workers, args.T)

    # measure SOPs
    print("Measuring SOPs (this runs a single pass over test set)...")
    sops_per_sample, total_samples = measure_sops_using_sopmonitor(model, test_loader, device, args)
    if sops_per_sample is None:
        print("[WARN] Could not measure SOPs because SOPMonitor not available. Please add utils.SOPMonitor or run profiling separately.")
    else:
        print(f"Measured avg SOPs per sample (sum across layers): {sops_per_sample:.4f}")

    # measure MACs via thop (optional)
    print("Attempting to measure MACs with thop (may fail for SNN wrappers)...")
    dummy_in = (args.batch_size, 3, 32, 32)
    macs = measure_macs_with_thop(model, (1,3,32,32), device)
    if macs is None:
        print("[INFO] MACs measurement with thop failed or not available.")
    else:
        macs_per_sample = macs
        print(f"MACs per sample (estimated by thop): {macs_per_sample:,}")

    # Constants
    E_sop = args.E_sop_pJ * 1e-12  # J per SOP
    E_mac = args.E_mac_pJ * 1e-12  # J per MAC

    # Compute inference energy per sample
    E_snn = (sops_per_sample * E_sop) if (sops_per_sample is not None) else 0.0
    E_ann = (macs_per_sample * E_mac) if ('macs_per_sample' in locals()) and macs_per_sample is not None else 0.0
    E_inf = E_snn + E_ann

    # training energy estimate
    train_mul = 1.0 + args.backward_mul
    E_train_per_sample = train_mul * E_inf
    dataset_size = len(dataset_test) * (int(5/1))  # dummy fallback — we will override below

    # best: infer training set size from torchvision CIFAR10 train set
    if args.dataset == 'CIFAR10':
        dataset_size = 50000
    elif args.dataset == 'CIFAR100':
        dataset_size = 50000

    steps_per_epoch = (dataset_size + args.batch_size - 1) // args.batch_size
    E_train_per_epoch = E_train_per_sample * dataset_size
    E_train_total = E_train_per_epoch * args.epochs

    # convert to easy units
    def fmt_j(x):
        if x >= 1.0:
            return f"{x:.6f} J"
        if x >= 1e-3:
            return f"{x*1e3:.4f} mJ"
        if x >= 1e-6:
            return f"{x*1e6:.3f} uJ"
        if x >= 1e-9:
            return f"{x*1e9:.3f} nJ"
        if x >= 1e-12:
            return f"{x*1e12:.3f} pJ"
        return f"{x:.3e} J"

    print("---- SATA-style energy estimates ----")
    print(f"E_sop (per SOP)   = {args.E_sop_pJ} pJ")
    print(f"E_mac (per MAC)   = {args.E_mac_pJ} pJ")
    print(f"SOPs per sample   = {sops_per_sample}")
    if 'macs_per_sample' in locals():
        print(f"MACs per sample   = {macs_per_sample:,}")
    print("Inference energy per sample:")
    print(f"  SNN component: {fmt_j(E_snn)}")
    print(f"  ANN component: {fmt_j(E_ann)}")
    print(f"  TOTAL (inference): {fmt_j(E_inf)}")
    print("Training estimate (approx):")
    print(f"  backward multiplier = {args.backward_mul} => per-sample train energy = {train_mul:.2f}x inference")
    print(f"  Per-epoch train energy (dataset_size={dataset_size}): {fmt_j(E_train_per_epoch)}")
    print(f"  Total train energy ({args.epochs} epochs): {fmt_j(E_train_total)}")

    # Save CSV
    out_row = {
        'checkpoint': args.checkpoint,
        'model': args.model,
        'dataset': args.dataset,
        'sops_per_sample': sops_per_sample if sops_per_sample is not None else '',
        'macs_per_sample': macs_per_sample if 'macs_per_sample' in locals() and macs_per_sample is not None else '',
        'E_sop_pJ': args.E_sop_pJ,
        'E_mac_pJ': args.E_mac_pJ,
        'E_inf_J': E_inf,
        'E_train_per_epoch_J': E_train_per_epoch,
        'E_train_total_J': E_train_total,
        'epochs': args.epochs,
        'dataset_size': dataset_size
    }

    header = list(out_row.keys())
    file_exists = os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(out_row)

    print("Wrote results to", args.out)
    return

if __name__ == '__main__':
    main()
