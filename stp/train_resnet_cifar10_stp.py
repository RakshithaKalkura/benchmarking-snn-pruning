'''
usage: python train_resnet_cifar10_stp.py --arch ResNet19 --dataset cifar10 --timestep 20 --batch_size 128 --learning_rate 0.001 --end_iter 100 --rewinding_epoch 20 --prune_iterations 5 --prune_percent 0.2 --round 1
'''
'''
this code is used to implement LTH on Spiking ResNet19 with CIFAR10 dataset with timestep decision based on KL divergence
spatio-temporal pruning method'''
# train_resnet19_cifar10_lth.py
import numpy as np
import torch
import utils
import config_lth
import torch.cuda.amp as amp
from torchvision import transforms
import torchvision
import os
import copy
import pickle
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from archs.cifar10.resnet import ResNet19  # make sure import path is correct (resnet19 used here)

from utils_for_snn_lth_temp_lamp import *   # make_mask, prune_by_percentile_weight_trace, original_initialization, make_mask, etc.
from utils import data_transforms  # optional if used elsewhere
from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based.functional import reset_net

# ---------------------------------------------------------------------
# Small helper: same TET_loss you used
def TET_loss(outputs, labels, criterion=nn.CrossEntropyLoss(), means=1.0, lamb=1e-3):
    outputs = torch.stack(outputs, dim=0) if not isinstance(outputs, torch.Tensor) else outputs
    # outputs may already be [T, N, C] as list->stack; ensure [T,N,C]
    if outputs.dim() == 3 and outputs.shape[0] == len(outputs):  # avoid double stacking
        pass
    # convert to [T, N, C] if list has been returned
    if isinstance(outputs, list):
        outputs = torch.stack(outputs, dim=0)
    # now permute to [N, T, C] then compute per-timestep loss as before
    outputs = outputs.permute(1, 0, 2)  # [N, T, C]
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd
# ---------------------------------------------------------------------

def find_decision(args, model, trainset, epoch=300):
    """Find an early timestep decision by KL divergence between full-T output and lower-T outputs"""
    model.eval()
    with torch.no_grad():
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=False, num_workers=4)
        # full-T outputs
        full_T = args.timestep
        out_full_list = []
        targets_full_list = []
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            out = model(inputs)           # model returns list [T, N, C] (or list length T)
            if isinstance(out, list):
                out_full = torch.stack(out, dim=0).mean(0)  # [N,C]
            else:
                out_full = torch.mean(out, dim=0)  # assuming shape [T,N,C]
            out_full_list.append(out_full)
            targets_full_list.append(F.one_hot(targets, 10).float())
            reset_net(model)
        out_full = torch.cat(out_full_list, dim=0)
        targets_full = torch.cat(targets_full_list, dim=0)

        candidate_t = [i for i in range(1, args.timestep + 1)]
        KDdivs = []
        for temp_t in candidate_t:
            out_temp_list = []
            for inputs, targets in train_loader:
                inputs = inputs.cuda()
                # feed only first temp_t timesteps: for static CIFAR10 we pass same input but model.total_timestep = temp_t
                model.total_timestep = temp_t
                out_temp = model(inputs)
                if isinstance(out_temp, list):
                    out_temp_agg = torch.stack(out_temp, dim=0).mean(0)
                else:
                    out_temp_agg = torch.mean(out_temp, dim=0)
                out_temp_list.append(out_temp_agg)
                reset_net(model)
            out_temp_all = torch.cat(out_temp_list, dim=0)
            # compute KL divergence to full_T outputs (use targets_full as proxy if you want)
            KDdiv = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(out_temp_all, dim=1),
                                                         F.softmax(out_full, dim=1))
            KDdivs.append(float(KDdiv.cpu().data.numpy()) * 1e4)
        # normalize
        KDarr = np.array(KDdivs)
        norm_KLdivs = ((KDarr - KDarr.min()) / (KDarr.max() - KDarr.min() + 1e-12)).tolist()
        threshold = 0.01
        new_timestep = args.timestep
        for i, n_kldiv in enumerate(norm_KLdivs):
            if n_kldiv < threshold:
                new_timestep = candidate_t[i]
                break
    print('Find decision timestep-------')
    print('epoch', epoch, ':', norm_KLdivs, '| New timestep', new_timestep)
    # restore total_timestep
    model.total_timestep = full_T
    return new_timestep

# ---------------------------------------------------------------------
def load_data_cifar10(data_dir, batch_size, workers=4, augment=False):
    """Load CIFAR-10 train/test DataLoaders (standard)"""
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return trainset, testset, train_loader, test_loader

# ---------------------------------------------------------------------
def test(model, test_loader, criterion, timestep):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.cuda(), targets.cuda()
            # make sure model uses desired timestep
            model.total_timestep = timestep
            out = model(imgs)
            # out is list length T, aggregate
            if isinstance(out, list):
                logits = torch.stack(out, dim=0).mean(0)  # [N,C]
            else:
                logits = torch.mean(out, dim=0)
            reset_net(model)
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    acc = 100.0 * correct / total
    return acc

# ---------------------------------------------------------------------
def train(args, epoch, train_loader, model, criterion, optimizer, scheduler=None, iteration=None, timestep=None):
    model.train()
    EPS = 1e-6
    train_loss = 0.0
    train_samples = 0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, labels = imgs.cuda(), labels.cuda()
        # Ensure model uses timestep
        model.total_timestep = timestep if timestep is not None else args.timestep
        out_list = model(imgs)  # list of length T
        loss = TET_loss(out_list, labels)
        loss.backward()
        # freeze pruned weights' grads (if mask scheme uses zeros)
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data
                if len(tensor.size()) == 1:
                    continue
                grad_tensor = p.grad
                if grad_tensor is None:
                    continue
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor
        optimizer.step()
        reset_net(model)
        train_samples += labels.numel()
        train_loss += loss.item() * labels.numel()
    train_loss /= train_samples
    if scheduler is not None:
        scheduler.step()
    return train_loss

# ---------------------------------------------------------------------
def main():
    args = config_lth.get_args()
    # ensure CLI args include: seed, timestep, batch_size, learning_rate, end_iter, rewinding_epoch, prune_iterations, prune_percent, round, resume_iter, resume_epoch, scheduler, print_freq, valid_freq
    np.random.seed(args.seed); random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # writer
    writer = SummaryWriter(f"./runs/resnet19_cifar10_decision/{args.dataset}_ResNet19")

    # load CIFAR-10
    trainset, testset, train_loader, test_loader = load_data_cifar10(args.data_dir, args.batch_size, workers=4, augment=True)

    # create model
    model = ResNet19(num_classes=10, total_timestep=args.timestep).cuda()

    # optionally load a checkpoint to initialize (if you keep the same pattern)
    # ckpt = torch.load('<path_to_initial_ckpt>'); model.load_state_dict(ckpt['net'])

    functional.set_backend(model, 'cupy')  # if cupy backend available; else remove

    # initializations
    initial_state_dict = copy.deepcopy(model.state_dict())
    timestep = find_decision(args, model, trainset)

    utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/resnet19_decision/{args.arch}/{args.dataset}/init_rewind")
    mask = make_mask(model)  # from your utils
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)

    all_loss = np.zeros(args.end_iter, float)
    all_accuracy = np.zeros(args.end_iter, float)
    Trace = {}

    # resume logic preserved (if you use resume_iter/resume_epoch)
    if getattr(args, 'resume_iter', 0) != 0:
        ckpt = torch.load(f"{os.getcwd()}/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/itearation{args.resume_iter - 1}/ckpt.pth.tar")
        model.load_state_dict(ckpt['net'])
        mask = ckpt['mask']
        bestacc = ckpt['bestacc']
        comp = ckpt['comp']
        _ite = ckpt['_ite']
        initial_state_dict = torch.load(f"{os.getcwd()}/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/init_rewind/rewind_state_dict.pth.tar")

    for _ite in range(getattr(args, 'resume_iter', 0), ITERATION):
        if not _ite == 0:
            # pruning & rewind
            rewinding_epoch = args.rewinding_epoch
            model, mask = prune_by_percentile_weight_trace(args, args.prune_percent, mask, model, Trace)
            model = original_initialization(mask, initial_state_dict, model)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        print(f"\n--- Pruning Level [round{args.round}:{_ite}/{ITERATION}]: ---")
        if getattr(args, 'scheduler', None) is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.end_iter - getattr(rewinding_epoch, 0)), eta_min=0)

        comp1, param_each_layer = utils.print_nonzeros(model)
        comp[_ite] = comp1
        writer.add_scalar('global_rate', comp1, _ite)
        for key, value in param_each_layer.items():
            writer.add_scalar(key + str(value[1]), round(value[0] / value[1], 2), _ite)

        utils.checkdir(f"{os.getcwd()}/dumps/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/itearation{_ite}")
        with open(f"{os.getcwd()}/dumps/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/itearation{_ite}/pruned{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)

        loss = 0
        accuracy = 0
        for iter_ in range(getattr(args, 'resume_epoch', 0) + 1, args.end_iter - getattr(rewinding_epoch, 0) + 1):
            if (iter_) % args.valid_freq == 0 or iter_ == 1:
                accuracy = test(model, test_loader, nn.CrossEntropyLoss(), timestep)
                writer.add_scalar('accuracy_' + f'{_ite}', accuracy, iter_ + getattr(rewinding_epoch, 0))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/itearation{_ite}")
                    torch.save(model, f"{os.getcwd()}/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/itearation{_ite}/pruned{1 - comp1}_acc{accuracy}_epoch{iter_}_model.pth.tar")

            # train for one epoch
            loss = train(args, iter_, train_loader, model, nn.CrossEntropyLoss(), optimizer, scheduler, _ite, timestep)
            all_loss[iter_ - 1] = loss
            all_accuracy[iter_ - 1] = accuracy

            if _ite == 0 and iter_ == args.rewinding_epoch:
                print('find laterewinding weight--------')
                initial_state_dict = copy.deepcopy(model.state_dict())
                utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/init_rewind")
                torch.save(initial_state_dict, f"{os.getcwd()}/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/init_rewind/rewind_state_dict.pth.tar")

            if (iter_) % args.print_freq == 0 or iter_ == 1:
                print(f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')

        bestacc[_ite] = best_accuracy

        checkpoint = {
            "net": model.state_dict(),
            'mask': mask,
            '_ite': _ite,
            'bestacc': bestacc,
            'comp': comp,
            'timestep': timestep
        }
        utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/itearation{_ite}")
        torch.save(checkpoint, f"{os.getcwd()}/snn_laterewind_lth/resnet19/{args.arch}/{args.dataset}/itearation{_ite}/ckpt.pth.tar")
        best_accuracy = 0

    writer.close()

# ---------------------------------------------------------------------
if __name__ == '__main__':
    main()
