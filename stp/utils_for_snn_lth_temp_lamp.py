import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy

from spikingjelly.activation_based.functional import reset_net
from spikingjelly.activation_based import layer


def _is_prunable_module(m):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)
            or isinstance(m, layer.Linear) or isinstance(m, layer.Conv2d))


def get_weights(model):
    weights = []
    for m in model.modules():
        if _is_prunable_module(m):
            weights.append(m.weight)
    return weights


def get_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(m)
    return modules


def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(torch.count_nonzero(m.weight))
    return torch.FloatTensor(unmaskeds)


def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp) - 1]
    # normalize by cumulative sum (avoid divide by zero)
    denom = (scores.sum() - scores_cumsum)
    # prevent tiny denominators
    denom[denom == 0] = 1e-12
    sorted_scores = sorted_scores / denom
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores
    return new_scores.view(scores.shape)


def _compute_lamp_amounts(model, amount):
    """
    Compute normalization schemes.
    Returns per-layer prune fraction in [0,1] (amounts).
    """
    unmaskeds = _count_unmasked_weights(model)
    total_survive = int(np.round(unmaskeds.sum().item() * (1.0 - amount)))

    # build concatenated normalized squared-score vector
    flattened_scores = [_normalize_scores(w ** 2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores, dim=0)

    if total_survive <= 0:
        # nothing survives -> keep tiny fraction
        total_survive = max(1, int(0.01 * concat_scores.numel()))

    topks, _ = torch.topk(concat_scores, total_survive)
    threshold = topks[-1]

    final_survs = [torch.ge(score, threshold * torch.ones(score.size(), device=score.device)).sum()
                   for score in flattened_scores]

    amounts = []
    for idx, final_surv in enumerate(final_survs):
        # ensure final_surv and unmaskeds[idx] are floats
        amounts.append(1.0 - (final_surv.float().item() / (unmaskeds[idx].float().item() + 1e-12)))
    return amounts


def make_mask(model):
    """
    Create a list of numpy masks (1/0 arrays) in the order of model.named_parameters()
    filtered by 'weight' in name. Call this after a warm-up forward pass.
    """
    mask = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask.append(np.ones_like(tensor, dtype=np.int8))
    return mask


# Prune by Percentile module (LAMP-style amounts)
def prune_by_percentile_weight_trace(args, percent, mask, model, trace):
    """
    percent: e.g., 20 for 20% -> will compute amounts via LAMP and prune that fraction
    mask: list of numpy masks (same order as make_mask)
    model: torch model
    trace: unused placeholder (kept to match existing API)
    """
    model_step = 0
    amount_step = 0
    amounts = _compute_lamp_amounts(model, percent * 0.01)

    for name, param in model.named_parameters():
        # Skip non-weights (bias, bn) and 1-D params
        if 'weight' in name:
            tensor_re = param.data.cpu().numpy()
            if tensor_re.ndim == 1:
                model_step += 1
                continue

            tensor_abs = np.abs(tensor_re)
            alive = tensor_abs[np.nonzero(tensor_abs)]

            if alive.size == 0:
                # nothing to prune in this layer
                model_step += 1
                amount_step += 1
                continue

            # safe index k for percentile selection (handles tiny layers)
            surv_frac = max(0.0, min(1.0, 1.0 - amounts[amount_step]))
            k = int(round(alive.shape[0] * surv_frac)) - 1
            k = max(0, min(k, alive.shape[0] - 1))
            percentile_value = np.sort(alive)[k]

            weight_dev = param.device
            # build new mask (numpy) consistent dtype
            new_mask = np.where(tensor_abs < percentile_value, 0, 1).astype(np.int8)
            # apply mask to parameter values
            param.data = torch.from_numpy(tensor_re * new_mask).to(weight_dev)
            mask[model_step] = new_mask

            model_step += 1
            amount_step += 1

    return model, mask


def get_pruning_maks(args, percent, mask, model):
    """
    Compute global percentile across all weights (ignoring 1-D params).
    Returns updated mask list (numpy arrays).
    """
    all_param = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if tensor.ndim == 1:
                continue
            alive = tensor[np.nonzero(tensor)]
            if alive.size > 0:
                all_param.append(np.abs(alive).ravel())

    if len(all_param) == 0:
        return mask

    param_whole = np.concatenate(all_param)
    k = int(float(param_whole.shape[0]) / float(100.0 / percent))
    k = max(0, min(k, param_whole.shape[0] - 1))
    percentile_value = np.sort(param_whole)[k]

    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if tensor.ndim == 1:
                step += 1
                continue
            new_mask = np.where(np.abs(tensor) < percentile_value, 0, 1).astype(np.int8)
            mask[step] = new_mask
            step += 1
    return mask


def original_initialization(mask_temp, initial_state_dict, model):
    """
    Reinitialize model weights from initial_state_dict but keep masked-out weights zeroed.
    mask_temp: list of numpy masks, same order as make_mask()
    initial_state_dict: a state_dict (torch) from initial weights
    """
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            # ensure correct shapes and types
            init_w = initial_state_dict[name].cpu().numpy()
            m = mask_temp[step]
            param.data = torch.from_numpy(m.astype(init_w.dtype) * init_w).to(weight_dev)
            step += 1
        if "bias" in name:
            param.data = initial_state_dict[name].to(param.device)
    return model


def original_initialization_nobias(mask_temp, initial_state_dict, model):
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            init_w = initial_state_dict[name].cpu().numpy()
            m = mask_temp[step]
            param.data = torch.from_numpy(m.astype(init_w.dtype) * init_w).to(weight_dev)
            step += 1
        if "bias" in name:
            # preserve bias but offset by +1 (original behavior)
            param.data = initial_state_dict[name].to(param.device) + 1
    return model


def test_dvs(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_samples = 0
    test_acc = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out_fr = model(data).mean(0)
            test_samples += label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            reset_net(model)
        test_acc /= test_samples
    return test_acc * 100.0


def test_dvs_timestep(model, test_loader, criterion, timestep=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_samples = 0
    test_acc = 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data[:, :timestep, :, :, :]
            data, label = data.to(device), label.to(device)
            out_fr = model(data).mean(0)
            test_samples += label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            reset_net(model)
        test_acc /= test_samples
    return test_acc * 100.0


def test_ann(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)
        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
    return accuracy


def weight_init(m):
    """
    Usage:
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)
