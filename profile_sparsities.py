#!/usr/bin/env python3
"""
profile_sparsities.py
Profiles a trained SNN to produce S, Gf, Gu per-layer in formats SATA expects.
"""
import argparse, os, yaml
import torch, numpy as np
from collections import defaultdict
from torchvision import datasets, transforms
# import your model builder here, e.g. from models.resnet import ResNet19SNN
# from your code import build_model, dataset_loader

def build_dataloader(batch_size, n_profile, dataset_root):
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    # use a small subset if n_profile less than dataset
    sampler = torch.utils.data.SubsetRandomSampler(list(range(min(n_profile, len(ds)))))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    return loader

def make_layer_list_and_yaml(model, dummy_input, out_path):
    # Run a forward pass and record shapes per conv/linear module in order.
    layer_entries = []
    hook_handles = []
    mod_list = []
    def fhook(mod, inp, out):
        mod_list.append((mod, out.shape))
    # register on conv and linear modules
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hook_handles.append(m.register_forward_hook(fhook))
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    for h in hook_handles:
        h.remove()
    # build yaml
    # naive naming: conv1, conv2, ..., fcN
    layers = []
    conv_idx = 1
    fc_idx = 1
    for (mod, shape) in mod_list:
        if isinstance(mod, torch.nn.Conv2d):
            in_ch = mod.in_channels; out_ch = mod.out_channels
            k_h, k_w = mod.kernel_size
            stride = mod.stride
            padding = mod.padding
            out_h, out_w = shape[2], shape[3]
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
        elif isinstance(mod, torch.nn.Linear):
            in_f = mod.in_features; out_f = mod.out_features
            layers.append({
                'name': f'fc{fc_idx}',
                'type': 'linear',
                'in_features': int(in_f),
                'out_features': int(out_f)
            })
            fc_idx += 1
    yaml_dict = {
        'name': 'resnet19_cifar10',
        'input': {'channels': 3, 'height': 32, 'width': 32},
        'batch_size': args.batch_size,
        'timesteps': args.T,
        'layers': layers
    }
    with open(os.path.join(out_path, 'resnet19_cifar10.yaml'), 'w') as f:
        yaml.dump(yaml_dict, f)
    return [l['name'] for l in layers]

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model (user must adapt this)
    model = build_model()  # change this for respective model 
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()

    loader = build_dataloader(args.batch_size, args.n_profile, args.dataset_root)

    # dummy input for YAML creation (assumes model accepts input with T handled inside model)
    dummy = torch.randn((1, 3, 32, 32), device=device)
    layer_names = make_layer_list_and_yaml(model, dummy, args.outdir)  # returns names in order

    # init accumulators
    accum = {ln: defaultdict(int) for ln in layer_names}
    param_to_layer = {}  # map param name -> layer_name (simple heuristic)
    for name, p in model.named_parameters():
        # attempt simple mapping by substring match
        for ln in layer_names:
            if ln in name:
                param_to_layer[name] = ln
                break
        if name not in param_to_layer:
            # fallback to first conv/linear
            param_to_layer[name] = layer_names[0]

    # Use hooks to gather activations and their grads
    saved_acts = {}
    saved_act_grads = defaultdict(list)
    handles = []
    modules = []
    for module in model.modules():
        # match modules to layer_names by order
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            modules.append(module)
            def make_fwd_hook(mod):
                def hook(mod_, inp, out):
                    # store output tensor (move to cpu later)
                    saved_acts[mod_] = out.detach().cpu()
                    # register grad hook to capture dL/d(out)
                    def act_grad_hook(grad):
                        saved_act_grads[mod_].append(grad.detach().cpu())
                    out.register_hook(act_grad_hook)
                return hook
            handles.append(module.register_forward_hook(make_fwd_hook(module)))

    criterion = torch.nn.CrossEntropyLoss()
    n_seen = 0
    for batch_idx, (x, y) in enumerate(loader):
        if n_seen >= args.n_profile:
            break
        x = x.to(device); y = y.to(device)
        # Forward/backward
        model.zero_grad()
        outputs = model(x)  # adapt if model expects T param: model(x, T=args.T)
        loss = criterion(outputs, y)
        loss.backward()

        # collect activations stats
        for mod in modules:
            if mod not in saved_acts: continue
            act = saved_acts[mod]  # cpu tensor
            nonzero = (act != 0).sum().item()
            accum_name = layer_names.pop(0) if False else None # we cannot map module->name easily here
            # Instead map by registering modules and layer_names in same order: assume same order
            # For simplicity, build a mapping earlier. (ensure mapping is correct.)
            # We'll instead map by sequential index:
        # weight grads
        for pname, p in model.named_parameters():
            ln = param_to_layer.get(pname, layer_names[0])
            if p.grad is not None:
                accum[ln]['nonzero_wgrad'] += (p.grad.detach().cpu() != 0).sum().item()
                accum[ln]['total_wgrad'] += p.numel()
        n_seen += x.size(0)

    # Remove hooks
    for h in handles: h.remove()

    # Postprocess and write CSV
    # You need to fill output_dim and weight_count per layer from YAML
    import math
    yaml_path = os.path.join(args.outdir, 'resnet19_cifar10.yaml')
    with open(yaml_path) as f:
        netspec = yaml.safe_load(f)
    layers = netspec['layers']
    layer_order = [l['name'] for l in layers]
    # Build S/Gf/Gu arrays
    S_arr, Gf_arr, Gu_arr = [], [], []
    rows = []
    for ln, lspec in zip(layer_order, layers):
        # default 0 if missing
        nonzero_spikes = accum[ln].get('nonzero_spikes', 0)
        total_spikes = accum[ln].get('total_spikes', 1)
        nonzero_dLdV = accum[ln].get('nonzero_dLdV', 0)
        total_dLdV = accum[ln].get('total_dLdV', 1)
        nonzero_wgrad = accum[ln].get('nonzero_wgrad', 0)
        total_wgrad = accum[ln].get('total_wgrad', 1)
        S = nonzero_spikes / total_spikes
        Gf = nonzero_dLdV / total_dLdV
        Gu = nonzero_wgrad / total_wgrad
        S_arr.append(S); Gf_arr.append(Gf); Gu_arr.append(Gu)
        if lspec['type'] == 'conv2d':
            out_dim = lspec['out_channels'] * lspec['out_h'] * lspec['out_w']
            wcount = lspec['in_channels'] * lspec['out_channels'] * lspec['kernel'][0] * lspec['kernel'][1]
        else:
            out_dim = lspec['out_features']
            wcount = lspec['in_features'] * lspec['out_features']
        rows.append([ln, S, Gf, Gu, out_dim, wcount])

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir,'S.npy'), np.array(S_arr))
    np.save(os.path.join(args.outdir,'Gf.npy'), np.array(Gf_arr))
    np.save(os.path.join(args.outdir,'Gu.npy'), np.array(Gu_arr))
    with open(os.path.join(args.outdir,'sparsity_stats.csv'), 'w') as f:
        f.write('layer,S,Gf,Gu,output_dim,weight_count\n')
        for r in rows:
            f.write(','.join([str(x) for x in r]) + '\n')

    print('Wrote outputs to', args.outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--n-profile', type=int, default=400)
    parser.add_argument('--dataset-root', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='./sata_inputs')
    args = parser.parse_args()
    main(args)
