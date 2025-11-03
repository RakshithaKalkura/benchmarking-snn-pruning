#!/usr/bin/env python3
"""
profile_sparsities.py

Usage example:
python profile_sparsities.py \
  --checkpoint ./checkpoints/checkpoint_max_acc1.pth \
  --batch-size 128 --T 10 --n-profile 400 --dataset-root ./data --outdir ./sata_inputs/unstructured

Outputs:
 - ./sata_inputs/resnet19_cifar10.yaml
 - ./sata_inputs/S.npy, Gf.npy, Gu.npy
 - ./sata_inputs/sparsity_stats.csv
"""
import argparse
import os
import yaml
from collections import defaultdict, OrderedDict
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn

# === USER: change this function to return your model instance ===
# The loader must construct the model (architecture) exactly as used for checkpoint
def build_model_for_repo(T=10, num_classes=10):
    # Example: your repo exposes resnet.ResNet19 or ResNet19SNN
    # from models.resnet import ResNet19  <-- adapt if needed
    # return ResNet19(num_classes=num_classes, total_timestep=T)
    # ---- default fallback: try common names from models package
    try:
        from models import resnet
        # try ResNet19SNN or ResNet19
        if 'ResNet19SNN' in resnet.__dict__:
            return resnet.__dict__['ResNet19SNN'](T=T, num_classes=num_classes)
        if 'ResNet19' in resnet.__dict__:
            return resnet.__dict__['ResNet19'](num_classes=num_classes, total_timestep=T)
    except Exception:
        pass
    raise RuntimeError("Please adapt build_model_for_repo() to construct your model.")

# === dataloader builder ===
def build_dataloader(batch_size, n_profile, dataset_root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    ds = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    # use first n_profile examples
    n = min(n_profile, len(ds))
    subset = torch.utils.data.Subset(ds, list(range(n)))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader

# === YAML generation and ordered module discovery ===
def make_layer_list_and_yaml(model, dummy_input, out_path, batch_size=128, T=10, yaml_name='resnet19_cifar10'):
    model.eval()
    module_list = []
    hook_handles = []

    # register temporary forward hooks on conv/linear to capture output shapes, in traversal order
    def fhook(mod, inp, out):
        # record module object and out shape
        module_list.append((mod, tuple(out.shape) if isinstance(out, torch.Tensor) else None))

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hook_handles.append(m.register_forward_hook(fhook))

    with torch.no_grad():
        _ = model(dummy_input)

    for h in hook_handles: h.remove()

    # build layer list in same order
    layers = []
    conv_idx = 1
    fc_idx = 1
    for (mod, shape) in module_list:
        if isinstance(mod, nn.Conv2d):
            in_ch = mod.in_channels; out_ch = mod.out_channels
            k_h, k_w = mod.kernel_size
            stride = mod.stride
            padding = mod.padding
            # Try to infer out_h/out_w from recorded shape; shape may be [T,N,C,H,W] or [N,C,H,W]
            if shape is not None:
                if len(shape) == 5:
                    out_h, out_w = shape[3], shape[4]
                    out_ch_recorded = shape[2]
                elif len(shape) == 4:
                    out_h, out_w = shape[2], shape[3]
                    out_ch_recorded = shape[1]
                else:
                    out_h, out_w = 1, 1
                    out_ch_recorded = out_ch
            else:
                out_h, out_w = 1, 1
                out_ch_recorded = out_ch
            layers.append({
                'name': f'conv{conv_idx}',
                'type': 'conv2d',
                'in_channels': int(in_ch),
                'out_channels': int(out_ch),
                'kernel': [int(k_h), int(k_w)],
                'stride': [int(stride[0]), int(stride[1])],
                'padding': [int(padding[0]), int(padding[1])],
                'out_h': int(out_h),
                'out_w': int(out_w)
            })
            conv_idx += 1
        elif isinstance(mod, nn.Linear):
            in_f = mod.in_features; out_f = mod.out_features
            layers.append({
                'name': f'fc{fc_idx}',
                'type': 'linear',
                'in_features': int(in_f),
                'out_features': int(out_f)
            })
            fc_idx += 1

    yaml_dict = {
        'name': yaml_name,
        'input': {'channels': 3, 'height': 32, 'width': 32},
        'batch_size': int(batch_size),
        'timesteps': int(T),
        'layers': layers
    }
    os.makedirs(out_path, exist_ok=True)
    yaml_path = os.path.join(out_path, f'{yaml_name}.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f)
    # Return ordered module objects and names (one-to-one)
    layer_names = [l['name'] for l in layers]
    # Re-query the modules in the same traversal order used earlier to get consistent mapping
    ordered_modules = [m for (m, _) in module_list if isinstance(m, (nn.Conv2d, nn.Linear))]
    return ordered_modules, layer_names, yaml_path

# === main profiling routine ===
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build model
    model = build_model_for_repo(T=args.T, num_classes=10)
    # load checkpoint (support common formats)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # remove module. prefix if present
    new_sd = {}
    for k, v in sd.items():
        nk = k[len('module.'):] if k.startswith('module.') else k
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()

    loader = build_dataloader(args.batch_size, args.n_profile, args.dataset_root)

    # make YAML and get ordered modules + layer names
    dummy = torch.randn((1, 3, 32, 32), device=device)
    ordered_modules, layer_names, yaml_path = make_layer_list_and_yaml(model, dummy, args.outdir,
                                                                       batch_size=args.batch_size, T=args.T)

    assert len(ordered_modules) == len(layer_names), "Module/layer count mismatch; inspect model YAML generation."

    # mapping: module -> layer_name (by index)
    mod2name = {mod: name for mod, name in zip(ordered_modules, layer_names)}

    # accumulators per layer name
    accum = {ln: defaultdict(int) for ln in layer_names}
    # also collect totals (for nonzero ratios)
    totals = {ln: defaultdict(int) for ln in layer_names}

    # Register forward hooks to capture activations and to attach gradient hooks on the outputs
    handles = []
    # map module->last_activation_tensor (cpu)
    last_act = {}
    # map module->list of dL/dOut tensors (cpu) per backward call (we only need nonzero statistics)
    grad_outs = {mod: [] for mod in ordered_modules}

    def make_fwd_hook(mod):
        def hook(mod_, inp, out):
            # out might be a list (SNN sometimes returns lists), or tensor with time dim
            out_cpu = out.detach().cpu()
            last_act[mod_] = out_cpu
            # register grad hook on the output tensor to capture dL/dOut
            def _grad_hook(grad):
                grad_outs[mod_].append(grad.detach().cpu())
            # if out is tensor, register hook
            if isinstance(out, torch.Tensor):
                out.register_hook(_grad_hook)
            # else if out is list/tuple, do not try to hook (rare). user must adapt.
        return hook

    for m in ordered_modules:
        handles.append(m.register_forward_hook(make_fwd_hook(m)))

    criterion = nn.CrossEntropyLoss()
    seen = 0
    for batch_idx, (imgs, targets) in enumerate(loader):
        if seen >= args.n_profile:
            break
        b = imgs.size(0)
        imgs = imgs.to(device)
        targets = targets.to(device)

        model.zero_grad()
        outputs = model(imgs)
        # if SNN returns a list of length T of tensors [N,C], convert to stacked tensor
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            logits = torch.stack(outputs, dim=0).mean(0)
        elif isinstance(outputs, torch.Tensor) and outputs.dim() == 3 and outputs.shape[0] == args.T:
            # shape [T, N, C]
            logits = outputs.mean(0)
        else:
            logits = outputs

        loss = criterion(logits, targets)
        loss.backward()

        # collect per-module stats
        for mod in ordered_modules:
            ln = mod2name[mod]
            # activation stats
            if mod in last_act:
                act = last_act[mod]  # cpu tensor
                # act shape could be [T,N,C,H,W] (SNN), [N,C,H,W] (ANN), [T,N,C] (pooled), or [N,C] for Linear
                # flatten sample dimension to count per-sample nonzeros
                # We'll compute total elements seen and nonzero elements across the batch
                # For S: fraction of non-zero activations = nonzero / total
                nonzero = (act != 0).sum().item()
                total = act.numel()
                accum[ln]['nonzero_spikes'] += nonzero
                totals[ln]['total_spikes'] += total

            # grad wrt output stats (Gf)
            glist = grad_outs.get(mod, [])
            if len(glist) > 0:
                # sum statistics across recorded grads for this batch
                nz = 0
                tt = 0
                for g in glist:
                    nz += (g != 0).sum().item()
                    tt += g.numel()
                accum[ln]['nonzero_dLdV'] += nz
                totals[ln]['total_dLdV'] += tt
            else:
                # maybe zero if activation didn't register grad hook
                pass

        # weight gradient stats
        for pname, p in model.named_parameters():
            # find corresponding layer name by prefix match (module name)
            # this is a fallback mapping: pick the first layer_name that is a substring of parameter name
            mapped = None
            for ln in layer_names:
                if ln in pname:
                    mapped = ln
                    break
            if mapped is None:
                # fallback: try sequential mapping by parameter order (not ideal)
                mapped = layer_names[0]
            if p.grad is not None:
                nz = (p.grad.detach().cpu() != 0).sum().item()
                accum[mapped]['nonzero_wgrad'] += nz
                totals[mapped]['total_wgrad'] += p.numel()
            else:
                totals[mapped]['total_wgrad'] += p.numel()

        seen += b
        # clear per-batch captures
        last_act.clear()
        for k in grad_outs: grad_outs[k].clear()

    # remove hooks
    for h in handles: h.remove()

    # read YAML (to compute dims/wcounts)
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)
    layers_spec = spec['layers']
    layer_order = [l['name'] for l in layers_spec]

    # finalize S,Gf,Gu arrays and CSV rows
    S_arr, Gf_arr, Gu_arr = [], [], []
    rows = []
    for ln, lspec in zip(layer_order, layers_spec):
        nonzero_spikes = accum[ln].get('nonzero_spikes', 0)
        total_spikes = totals[ln].get('total_spikes', 1)
        nonzero_dLdV = accum[ln].get('nonzero_dLdV', 0)
        total_dLdV = totals[ln].get('total_dLdV', 1)
        nonzero_wgrad = accum[ln].get('nonzero_wgrad', 0)
        total_wgrad = totals[ln].get('total_wgrad', 1)

        S = float(nonzero_spikes) / float(total_spikes) if total_spikes > 0 else 0.0
        Gf = float(nonzero_dLdV) / float(total_dLdV) if total_dLdV > 0 else 0.0
        Gu = float(nonzero_wgrad) / float(total_wgrad) if total_wgrad > 0 else 0.0

        S_arr.append(S); Gf_arr.append(Gf); Gu_arr.append(Gu)

        if lspec['type'] == 'conv2d':
            out_dim = lspec['out_channels'] * lspec['out_h'] * lspec['out_w']
            wcount = lspec['in_channels'] * lspec['out_channels'] * lspec['kernel'][0] * lspec['kernel'][1]
        else:
            out_dim = lspec.get('out_features', 1)
            wcount = lspec.get('in_features', 1) * lspec.get('out_features', 1)

        rows.append([ln, S, Gf, Gu, int(out_dim), int(wcount)])

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, 'S.npy'), np.array(S_arr))
    np.save(os.path.join(args.outdir, 'Gf.npy'), np.array(Gf_arr))
    np.save(os.path.join(args.outdir, 'Gu.npy'), np.array(Gu_arr))
    with open(os.path.join(args.outdir, 'sparsity_stats.csv'), 'w') as f:
        f.write('layer,S,Gf,Gu,output_dim,weight_count\n')
        for r in rows:
            f.write(','.join([str(x) for x in r]) + '\n')

    print('Wrote outputs to', args.outdir)
    print('YAML:', yaml_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--n-profile', type=int, default=400)
    p.add_argument('--dataset-root', type=str, default='./data')
    p.add_argument('--outdir', type=str, default='./sata_inputs')
    args = p.parse_args()
    main(args)
