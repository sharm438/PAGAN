import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import models as models
import torch.nn as nn
import math
import random

from collections import Counter, defaultdict
from copy import deepcopy

import pdb
import json

from typing import List
from sklearn.model_selection import train_test_split

def _make_local_splits(distributed_data, distributed_label, local_test_frac, local_val_frac, seed, print_stats=False):
    """
    Splits data into Train, Test, and optional Validation sets using stratified sampling.
    
    Args:
        local_test_frac: Fraction of total data for Test.
        local_val_frac: Fraction of total data for Validation.
    """
    # Basic Input Validation
    if not (0.0 <= local_test_frac < 1.0) or not (0.0 <= local_val_frac < 1.0):
        raise ValueError("Fractions must be between 0 and 1.")
    
    train_data, train_label = [], []
    val_data,   val_label   = [], []
    test_data,  test_label  = [], []

    for node_idx, (x, y) in enumerate(zip(distributed_data, distributed_label)):
        n_total = x.shape[0]
        
        # 1. Identify "Rare" classes (< 2 samples) to force into Train
        # We cannot stratify split classes with 1 sample.
        unique, counts = np.unique(y.numpy(), return_counts=True)
        rare_classes = unique[counts < 2]
        
        if len(rare_classes) > 0:
            # Indices of rare items
            rare_mask = np.isin(y.numpy(), rare_classes)
            safe_mask = ~rare_mask
            
            x_rare, y_rare = x[rare_mask], y[rare_mask]
            x_safe, y_safe = x[safe_mask], y[safe_mask]
        else:
            x_rare, y_rare = torch.tensor([]), torch.tensor([])
            x_safe, y_safe = x, y

        # 2. First Split: Separation into (Train+Val) and (Test)
        # If test_frac is 0, skip
        if local_test_frac > 0 and len(x_safe) > 0:
            x_tv, x_test, y_tv, y_test = train_test_split(
                x_safe, y_safe, 
                test_size=local_test_frac, 
                stratify=y_safe, 
                random_state=seed
            )
        else:
            x_tv, y_tv = x_safe, y_safe
            x_test, y_test = torch.tensor([]), torch.tensor([])

        # 3. Second Split: Separation into (Train) and (Val)
        # Note: val_frac is % of TOTAL. But we are splitting 'x_tv' which is (1 - test_frac) of total.
        # We need to adjust the fraction relative to x_tv.
        
        remaining_frac = 1.0 - local_test_frac
        if local_val_frac > 0 and remaining_frac > 0 and len(x_tv) > 0:
            relative_val_frac = local_val_frac / remaining_frac
            
            # Safety clip
            relative_val_frac = min(max(relative_val_frac, 0.0), 1.0)
            
            # Check for rare classes again in the remaining set before splitting
            # (Splitting might have reduced counts again, though unlikely with stratify)
            u_tv, c_tv = np.unique(y_tv.numpy(), return_counts=True)
            rare_tv = u_tv[c_tv < 2]
            
            if len(rare_tv) > 0:
                 # Force rare into train, split the rest
                 m_rare = np.isin(y_tv.numpy(), rare_tv)
                 x_tv_rare, y_tv_rare = x_tv[m_rare], y_tv[m_rare]
                 x_tv_safe, y_tv_safe = x_tv[~m_rare], y_tv[~m_rare]
                 
                 if len(x_tv_safe) > 0:
                     x_tr_safe, x_val, y_tr_safe, y_val = train_test_split(
                        x_tv_safe, y_tv_safe,
                        test_size=relative_val_frac,
                        stratify=y_tv_safe,
                        random_state=seed
                     )
                     # Combine safe train with rare train
                     x_train = torch.cat([x_tr_safe, x_tv_rare]) if len(x_tv_rare)>0 else x_tr_safe
                     y_train = torch.cat([y_tr_safe, y_tv_rare]) if len(y_tv_rare)>0 else y_tr_safe
                 else:
                     # All remaining were rare -> All to Train
                     x_train, y_train = x_tv, y_tv
                     x_val, y_val = torch.tensor([]), torch.tensor([])
            else:
                # Clean split
                x_train, x_val, y_train, y_val = train_test_split(
                    x_tv, y_tv,
                    test_size=relative_val_frac,
                    stratify=y_tv,
                    random_state=seed
                )
        else:
            x_train, y_train = x_tv, y_tv
            x_val, y_val = torch.tensor([]), torch.tensor([])

        # 4. Add the initial rare items back to TRAIN
        # (We forced them out at step 1)
        if len(x_rare) > 0:
            # Need to cast type to ensure compatibility if concatenation is needed
            # Assuming tensors are compatible
            if isinstance(x_train, torch.Tensor) and x_train.numel() == 0:
                 x_train, y_train = x_rare, y_rare
            else:
                 x_train = torch.cat([x_train, x_rare])
                 y_train = torch.cat([y_train, y_rare])

        # Append to lists
        train_data.append(x_train)
        train_label.append(y_train)
        val_data.append(x_val)
        val_label.append(y_val)
        test_data.append(x_test)
        test_label.append(y_test)

    return {
        "train_data": train_data, "train_label": train_label,
        "val_data": val_data,     "val_label": val_label,
        "test_data": test_data,   "test_label": test_label
    }

# ----------------------------
# Containers
# ----------------------------
class TrainObject:
    def __init__(self, dataset_name, net_name, train_data, train_labels,
                 test_data, num_inputs, num_outputs, lr, batch_size):
        self.dataset_name = dataset_name
        self.net_name = net_name
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = lr
        self.batch_size = batch_size

class DistributedData:
    """
    Holds per-node train/val splits and weights.
    """
    def __init__(self,
                 train_input, train_output,
                 wts=None, label_distribution=None,
                 node_to_cluster=None):
        self.train_input = train_input
        self.train_output = train_output
        self.wts = wts
        self.label_distribution = label_distribution
        self.node_to_cluster = node_to_cluster  # list[int] of cluster id per node

## Added on 120925
from scipy.spatial.distance import jensenshannon

def compute_pairwise_jsd(label_distributions):
    """
    Computes the NxN matrix of Jensen-Shannon Divergences between label distributions.
    Input: label_distributions (np.array of shape [N, num_classes])
    Output: torch.Tensor of shape [N, N]
    """
    N = len(label_distributions)
    jsd_mat = np.zeros((N, N), dtype=np.float32)
    
    # Normalize distributions to sum to 1
    # Add epsilon to avoid division by zero
    dists = label_distributions.astype(float)
    row_sums = dists.sum(axis=1, keepdims=True)
    dists = dists / (row_sums + 1e-9)

    for i in range(N):
        for j in range(i + 1, N):
            # Base 2 is standard for JSD in bits, but e is fine too. 
            # Scipy uses base e by default if base is None.
            val = jensenshannon(dists[i], dists[j], base=2.0)
            jsd_mat[i, j] = val
            jsd_mat[j, i] = val
            
    return torch.tensor(jsd_mat)

# Add this to utils_1119.py

def evaluate_and_log(
    current_round,
    metrics,
    model_source,      # For P2P: dict {node_id: weights}. For FedAvg: the global_model object.
    mode,              # 'p2p' (includes isolated/pagan) or 'federated' (fedavg/fedsgd)
    test_data,         # Global test set
    local_test_pairs,  # List of (data, label) tuples or None
    device,
    batch_size,
    dataset_name,
    net_name=None, inp_dim=None, out_dim=None, # Required only if mode='p2p' to reconstruct models
    eval_local=False
):
    """
    Centralized evaluation loop. Updates metrics dict in-place and prints status.
    """
    
    # Prepare storage
    g_accs, g_losses = [], []
    l_accs, l_losses = [], []
    
    # Helper to get input shape for tracing
    input_shape = [batch_size] + get_input_shape(dataset_name)
    dummy_input = torch.randn(input_shape).to(device)

    # ---------------------------------------------------------
    # PATH A: FEDERATED (1 Global Model)
    # ---------------------------------------------------------
    if mode == 'federated':
        global_model = model_source
        global_model.eval()
        
        # 1. Trace once
        traced_model = torch.jit.trace(deepcopy(global_model), dummy_input)
        
        # 2. Global Test Evaluation
        loss, acc = evaluate_global_metrics(traced_model, test_data, device)
        g_accs = acc # Single value
        g_losses = loss
        
        print(f"[Round {current_round}] FedAvg => Global Acc: {acc:.4f}, Loss: {loss:.4f}")

        # 3. Local Test Evaluation (Same global model on N local datasets)
        if eval_local and local_test_pairs is not None:
            for node_id, (ldata, llabel) in enumerate(local_test_pairs):
                loader = make_loader_from_tensors(ldata, llabel, batch_size=batch_size, shuffle=False)
                l_loss, l_acc = evaluate_global_metrics(traced_model, loader, device)
                l_losses.append(l_loss)
                l_accs.append(l_acc)

    # ---------------------------------------------------------
    # PATH B: P2P / ISOLATED (N Local Models)
    # ---------------------------------------------------------
    else: # mode == 'p2p'
        node_weights = model_source
        num_nodes = len(node_weights)
        
        for node_id in range(num_nodes):
            # 1. Reconstruct & Trace Node Model
            node_model = vec_to_model(node_weights[node_id], net_name, inp_dim, out_dim, device)
            traced_model = torch.jit.trace(deepcopy(node_model), dummy_input)
            
            # 2. Global Test Evaluation (How well this node does on global data)
            loss, acc = evaluate_global_metrics(traced_model, test_data, device)
            g_accs.append(acc)
            g_losses.append(loss)
            
            # 3. Local Test Evaluation (Node i on Data i)
            if eval_local and local_test_pairs is not None:
                ldata, llabel = local_test_pairs[node_id]
                loader = make_loader_from_tensors(ldata, llabel, batch_size=batch_size, shuffle=False)
                l_loss, l_acc = evaluate_global_metrics(traced_model, loader, device)
                l_losses.append(l_loss)
                l_accs.append(l_acc)

        print(f"[Round {current_round}] {mode.upper()} => Local Acc range (global test): [{min(g_accs):.4f}, {max(g_accs):.4f}]")

    # ---------------------------------------------------------
    # COMMON: LOGGING
    # ---------------------------------------------------------
    metrics['round'].append(current_round)
    metrics['global_acc'].append(g_accs)
    metrics['global_loss'].append(g_losses)

    if eval_local and local_test_pairs is not None:
        metrics['local_acc'].append(l_accs)
        metrics['local_loss'].append(l_losses)
        # Print local stats
        valid_accs = [a for a in l_accs if not math.isnan(a)]
        if valid_accs:
            print_acc_values(valid_accs)
            
    return metrics # Optional return, as dict is updated in-place

def print_acc_values(accs):
    """
    Prints 5 statistics for a list of accuracies:
    a = Min accuracy
    b = Avg of bottom 20 percentile
    c = Avg of mid 60 percentile
    d = Avg of high 20 percentile
    e = Max accuracy
    """
    if not accs:
        print("           Local-Test Acc: (No valid local test sets)")
        return

    # 1. Sort the accuracies
    sorted_accs = sorted(accs)
    n = len(sorted_accs)

    # 2. Calculate cut indices
    # int() floors the value, ensuring we get strict 20% cuts
    cut_low = int(n * 0.2)
    cut_high = int(n * 0.8)

    # 3. Define slices
    bottom_20 = sorted_accs[:cut_low]
    mid_60 = sorted_accs[cut_low:cut_high]
    high_20 = sorted_accs[cut_high:]

    # 4. Calculate Stats
    # a: Min
    stat_min = sorted_accs[0]

    # b: Avg Bottom 20% (fallback to min if n < 5 and slice is empty)
    stat_bot = sum(bottom_20) / len(bottom_20) if bottom_20 else stat_min

    # c: Avg Mid 60% (fallback to global average if slice is empty)
    stat_mid = sum(mid_60) / len(mid_60) if mid_60 else (sum(sorted_accs) / n)

    # d: Avg High 20% (fallback to max if n < 5 and slice is empty)
    stat_high = sum(high_20) / len(high_20) if high_20 else sorted_accs[-1]

    # e: Max
    stat_max = sorted_accs[-1]
    
    stat_avg = sum(accs)/len(accs)

    # 5. Print
    print(
        f"           Local-Test Stats: "
        f"[Min: {stat_min:.3f}, "
        f"Low20%: {stat_bot:.3f}, "
        f"Mid60%: {stat_mid:.3f}, "
        f"Top20%: {stat_high:.3f}, "
        f"Max: {stat_max:.3f}, "
        f"Avg: {stat_avg:.3f}]"
    )

from scipy.spatial.distance import jensenshannon
def analyze_train_test_split_distribution(train_labels_list, test_labels_list, num_classes):
    """
    Returns per-node dicts: node_id, train_counts, test_counts, jsd,
    plus compact string versions 'train_counts_str'/'test_counts_str' for single-line JSON.
    """
    print("\n[Sanity Check] Analyzing local train/test split class distributions...")

    all_js_divergences = []
    split_stats = []

    for i, (train_labels, test_labels) in enumerate(zip(train_labels_list, test_labels_list)):
        if len(train_labels) == 0 or len(test_labels) == 0:
            continue

        train_counts = np.bincount(train_labels.cpu().numpy(), minlength=num_classes)
        test_counts  = np.bincount(test_labels.cpu().numpy(),  minlength=num_classes)

        # Normalize for JSD
        train_dist = train_counts / (train_counts.sum() if train_counts.sum() > 0 else 1)
        test_dist  = test_counts  / (test_counts.sum()  if test_counts.sum()  > 0 else 1)
        js_div = float(jensenshannon(train_dist, test_dist, base=2.0))
        all_js_divergences.append(js_div)

        # Compact strings for single-line JSON display
        train_str = json.dumps(train_counts.tolist(), separators=(',', ':'))
        test_str  = json.dumps(test_counts.tolist(),  separators=(',', ':'))

        split_stats.append({
            "node_id": i,
            "train_counts": train_counts.tolist(),
            "test_counts": test_counts.tolist(),
            "jsd": js_div,
            "train_counts_str": train_str,
            "test_counts_str": test_str
        })

    if all_js_divergences:
        avg_jsd = float(np.mean(all_js_divergences))
        max_jsd = float(np.max(all_js_divergences))
        min_jsd = float(np.min(all_js_divergences))
        print(f"  Summary across {len(all_js_divergences)} nodes with splits:")
        print(f"  Jensen-Shannon Divergence -> Avg: {avg_jsd:.4f}, Min: {min_jsd:.4f}, Max: {max_jsd:.4f}")
        print("  (JSD of 0 means identical distributions; higher values mean more dissimilar.)")
    else:
        print("  No nodes had both train and test splits to analyze.")
    print("-" * 30)

    return split_stats


# ----------------------------
# Dataset loaders
# ----------------------------
def load_data(dataset_name, batch_size, lr, fraction=1.0):
    """
    Loads a dataset fully into memory for train (tensors) and returns a standard global test DataLoader.
    """
    if dataset_name == 'mnist':
        num_inputs, num_outputs = 28*28, 10
        net_name = 'lenet'
        if batch_size is None: batch_size = 32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
        if fraction < 1.0:
            size = int(len(full_train)*fraction)
            full_train, _ = random_split(full_train, [size, len(full_train)-size])
        loader = torch.utils.data.DataLoader(full_train, batch_size=len(full_train), shuffle=False)
        data, labels = next(iter(loader))
        testset = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'cifar10':
        num_inputs, num_outputs = 32*32*3, 10
        net_name = 'cifar_cnn'
        if batch_size is None: batch_size = 128
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010))
        ])
        full_train = torchvision.datasets.CIFAR10('./data', train=True, transform=transform_train, download=True)
        if fraction < 1.0:
            size = int(len(full_train)*fraction)
            full_train, _ = random_split(full_train, [size, len(full_train)-size])
        loader = torch.utils.data.DataLoader(full_train, batch_size=len(full_train), shuffle=False)
        data, labels = next(iter(loader))
        testset = torchvision.datasets.CIFAR10('./data', train=False, transform=transform_test, download=True)
        test_data = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    elif dataset_name == 'tiny_imagenet':
        num_inputs, num_outputs = 64*64*3, 200
        net_name = 'resnet18'
        if batch_size is None: batch_size = 128
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        ])
        trainset = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/train', transform=transform_train)
        if fraction < 1.0:
            size = int(len(trainset)*fraction)
            trainset, _ = random_split(trainset, [size, len(trainset)-size])
        loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
        data, labels = next(iter(loader))
        testset = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'cifar100':
        num_inputs, num_outputs = 32*32*3, 100
        net_name = 'resnet50'
        if batch_size is None: batch_size = 128
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071,0.4865,0.4409),
                                 (0.2673,0.2564,0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071,0.4865,0.4409),
                                 (0.2673,0.2564,0.2761))
        ])
        full_train = torchvision.datasets.CIFAR100('./data', train=True, transform=transform_train, download=True)
        if fraction < 1.0:
            size = int(len(full_train)*fraction)
            full_train, _ = random_split(full_train, [size, len(full_train)-size])
        loader = torch.utils.data.DataLoader(full_train, batch_size=len(full_train), shuffle=False)
        data, labels = next(iter(loader))
        testset = torchvision.datasets.CIFAR100('./data', train=False, transform=transform_test, download=True)
        test_data = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    elif dataset_name == 'agnews':
        # Minimal, reproducible text loader using torchtext basic_english tokenizer
        from torchtext.datasets import AG_NEWS
        from torchtext.data.utils import get_tokenizer
        if batch_size is None: batch_size = 64
        train_iter, test_iter = AG_NEWS(split=('train', 'test'))
        tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        raw_train = []
        for (lbl, txt) in train_iter:
            raw_train.append((lbl, txt))
            counter.update(tokenizer(txt))
        vocab = list(counter.keys())
        stoi = {w: i+2 for i, w in enumerate(vocab)}  # 0:PAD, 1:UNK
        stoi["<PAD>"] = 0
        stoi["<UNK>"] = 1

        def to_ids(s):
            return [stoi.get(t, 1) for t in tokenizer(s)]

        # train tensors
        max_len = 0
        xs, ys = [], []
        for (y, s) in raw_train:
            ids = to_ids(s)
            max_len = max(max_len, len(ids))
            xs.append(ids)
            ys.append(y-1)
        def pad(z, L): return z[:L] if len(z) >= L else z + [0]*(L-len(z))
        data = torch.stack([torch.tensor(pad(z, max_len), dtype=torch.long) for z in xs])
        labels = torch.tensor(ys, dtype=torch.long)

        # test tensors
        raw_test = list(test_iter)
        xs_t = [to_ids(s) for (_, s) in raw_test]
        ys_t = [y-1 for (y, _) in raw_test]
        Xtest = torch.stack([torch.tensor(pad(z, max_len), dtype=torch.long) for z in xs_t])
        Ytest = torch.tensor(ys_t, dtype=torch.long)
        test_ds = torch.utils.data.TensorDataset(Xtest, Ytest)
        test_data = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        num_inputs = max_len
        num_outputs = 4
        net_name = 'agnews_net'

    elif dataset_name == 'femnist':
        # FEMNIST loader (users -> combined train tensors, global test)
        import glob, json
        from PIL import Image
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        def read_json_dir(path):
            dist_X, dist_y = [], []
            X_all, y_all = [], []
            for fp in sorted(glob.glob(f"{path}/*.json")):
                with open(fp, 'r') as f:
                    obj = json.load(f)
                for user in obj['users']:
                    ux = obj['user_data'][user]['x']
                    uy = obj['user_data'][user]['y']
                    imgs = [transform(Image.fromarray(np.array(img).reshape(28, 28).astype(np.float32))) for img in ux]
                    imgs = torch.stack(imgs)
                    yy = torch.tensor(uy, dtype=torch.long)
                    dist_X.append(imgs)
                    dist_y.append(yy)
                    X_all.extend(list(imgs))
                    y_all.extend(list(yy))
            return dist_X, dist_y, torch.stack(X_all), torch.tensor(y_all)
        train_path = './data/femnist_700/data/train'
        test_path  = './data/femnist_700/data/test'
        dist_X, dist_y, X_all, y_all = read_json_dir(train_path)
        if fraction < 1.0:
            total = len(X_all)
            idx = torch.randperm(total)[:int(fraction*total)]
            X_all = X_all[idx]; y_all = y_all[idx]
        _, _, Xtest_all, Ytest_all = read_json_dir(test_path)
        test_ds = torch.utils.data.TensorDataset(Xtest_all, Ytest_all)
        if batch_size is None: batch_size = 32
        test_data = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        dataset_name = 'femnist'
        net_name = 'femnist_cnn'
        num_inputs = 28*28
        num_outputs = 62
        data, labels = X_all, y_all

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return TrainObject(dataset_name, net_name, data, labels, test_data,
                       num_inputs, num_outputs, lr, batch_size)


# ----------------------------
# Clustered heterogeneity distributor
# ----------------------------
def clustered_distribute_data(
    data_tuple,
    num_nodes: int,
    num_clusters: int,
    alpha: float,                 # Dirichlet α (small => more non-IID)
    out_dim: int,
    device: torch.device,
    seed: int = 42
) -> DistributedData:
    """
    Stage 1: Capacity-aware per-class Dirichlet across CLUSTERS with exact cluster totals (~ N/num_clusters).
    Stage 2: IID equal-chunk split within each cluster across its nodes.
    Returns: per-node TRAIN tensors only (no val), FedAvg weights, label histogram, node->cluster map.
    If any node would be empty, prints debug info and raises ValueError.
    """
    X, y = data_tuple
    N = len(X)
    #print(torch.bincount(y.cpu()).tolist())
    
    assert N == len(y), "X and y length must match."
    assert num_nodes > 0 and num_clusters > 0, "num_nodes/num_clusters must be > 0."
    alpha = float(max(alpha, 1e-8))

    # Seeding
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # ---- Per-class index pools
    class_indices = [[] for _ in range(out_dim)]
    for i, lbl in enumerate(y):
        class_indices[int(lbl.item())].append(i)
    class_sizes = np.array([len(ix) for ix in class_indices], dtype=int)

    # ---- Nodes per cluster (near-equal)
    base = num_nodes // num_clusters
    r = num_nodes % num_clusters
    nodes_per_cluster = [base + (1 if i < r else 0) for i in range(num_clusters)]
    node_to_cluster = []
    for cid, cnt in enumerate(nodes_per_cluster):
        node_to_cluster.extend([cid] * cnt)

    # ---- Exact target totals per cluster (as equal as possible)
    target_capacity = np.full(num_clusters, N // num_clusters, dtype=int)
    target_capacity[: (N % num_clusters)] += 1
    residual_capacity = target_capacity.copy()

    # ---- Shuffle each class index list once (deterministic slicing)
    for cls in range(out_dim):
        rng.shuffle(class_indices[cls])

    # ---- Stage 1: capacity-aware per-class Dirichlet
    # cluster_class_counts[cid][cls]
    cluster_class_counts = [np.zeros(out_dim, dtype=int) for _ in range(num_clusters)]
    for cls in range(out_dim):
        M = int(class_sizes[cls])
        if M == 0:
            continue

        cap = residual_capacity.astype(float)
        # If no capacity left globally (shouldn't happen before final classes), skip
        if cap.sum() <= 0:
            continue

        p = rng.dirichlet([alpha] * num_clusters)  # preference over clusters for this class
        # Capacity-weighted desire, then scaled to class size M
        weights = p * cap
        # If weights degenerate (all zeros), fall back to pure capacity
        if weights.sum() <= 0:
            weights = cap
        weights /= weights.sum()

        raw = weights * M  # desired fractional counts per cluster (capacity-aware)
        # Floor, but do not exceed residual capacity
        alloc = np.minimum(np.floor(raw).astype(int), residual_capacity)

        assigned = int(alloc.sum())
        leftover = M - assigned
        if leftover > 0:
            # Distribute leftover by largest fractional parts among clusters that still have capacity
            frac = raw - alloc
            # invalidate clusters that have no room
            no_room = (residual_capacity - alloc) <= 0
            frac[no_room] = -1e9
            order = np.argsort(-frac)
            for cid in order:
                if leftover == 0:
                    break
                room = int(residual_capacity[cid] - alloc[cid])
                if room > 0:
                    take = min(room, leftover)
                    alloc[cid] += take
                    leftover -= take

        # Final guard: if still leftover (extremely rare), fill any slot with capacity
        if leftover > 0:
            for cid in range(num_clusters):
                if leftover == 0:
                    break
                room = int(residual_capacity[cid] - alloc[cid])
                if room > 0:
                    take = min(room, leftover)
                    alloc[cid] += take
                    leftover -= take

        # Commit this class
        if alloc.sum() != M:
            raise RuntimeError(f"Allocator bug: class {cls} M={M} alloc={int(alloc.sum())}")

        for cid in range(num_clusters):
            take = int(alloc[cid])
            cluster_class_counts[cid][cls] += take
            residual_capacity[cid] -= take

        # Strong invariant: never exceed capacity
        if np.any(residual_capacity < 0):
            raise RuntimeError("Exceeded cluster capacity; allocator must be fixed.")

    # After all classes, capacities must be exactly filled
    if np.any(residual_capacity != 0):
        # Print helpful debug and fail fast
        print("[ClusteredSplit:ERROR] Residual capacities not zero:",
              residual_capacity.tolist())
        print("  target_capacity:", target_capacity.tolist())
        print("  per-class sizes:", class_sizes.tolist())
        print("  per-cluster totals:",
              [int(cc.sum()) for cc in cluster_class_counts])
        raise ValueError("Cluster totals did not match targets; aborting.")

    # ---- Build index lists per cluster from counts
    per_class_ptr = [0] * out_dim
    cluster_indices = [[] for _ in range(num_clusters)]
    for cid in range(num_clusters):
        for cls in range(out_dim):
            k = int(cluster_class_counts[cid][cls])
            if k > 0:
                start = per_class_ptr[cls]; end = start + k
                cluster_indices[cid].extend(class_indices[cls][start:end])
                per_class_ptr[cls] = end
        rng.shuffle(cluster_indices[cid])  # small intra-cluster shuffle

    # ---- Stage 2: split clusters across their nodes (IID equal chunks)
    train_input: List[torch.Tensor] = [None] * num_nodes
    train_output: List[torch.Tensor] = [None] * num_nodes

    node_base_offsets = np.cumsum([0] + nodes_per_cluster[:-1]).tolist()
    for cid in range(num_clusters):
        c_nodes = nodes_per_cluster[cid]
        c_idx = cluster_indices[cid]
        c_size = len(c_idx)

        if c_nodes == 0:
            continue
        if c_size < c_nodes:
            print(f"[ClusteredSplit:ERROR] Cluster {cid} size {c_size} < nodes {c_nodes}.")
            raise ValueError("Cannot split without creating empty nodes.")

        # equal-chunk sizes
        sizes = [c_size // c_nodes] * c_nodes
        for i in range(c_size % c_nodes):
            sizes[i] += 1

        start = 0
        base_idx = node_base_offsets[cid]
        for j in range(c_nodes):
            end = start + sizes[j]
            shard = c_idx[start:end]
            if end <= start:
                print(f"[ClusteredSplit:ERROR] Empty shard for node in cluster {cid}.")
                raise ValueError("Empty node shard; aborting.")

            node_idx = base_idx + j
            x_shard = torch.stack([X[k] for k in shard]).to(device)
            y_shard = torch.tensor([int(y[k].item()) for k in shard], dtype=torch.long, device=device)

            train_input[node_idx]  = x_shard
            train_output[node_idx] = y_shard
            start = end

    # Final safety: ensure every node got data
    none_nodes = [i for i, t in enumerate(train_input) if t is None]
    if none_nodes:
        print("[ClusteredSplit:ERROR] Nodes with no data:", none_nodes)
        raise ValueError("One or more nodes received no data; aborting.")

    # ---- FedAvg weights and label histogram
    sizes = torch.tensor([len(t) for t in train_input], dtype=torch.float32, device=device)
    wts = sizes / sizes.sum()

    label_distribution = np.zeros((num_nodes, out_dim), dtype=int)
    for i in range(num_nodes):
        for yy in train_output[i].detach().cpu().tolist():
            label_distribution[i][yy] += 1

    # ---- Debug summaries
    _print_node_size_stats([int(s.item()) for s in sizes])
    cluster_sizes = [len(cluster_indices[cid]) for cid in range(num_clusters)]
    print(f"[ClusteredSplit] α={alpha:.4g} | cluster totals (min/med/max) = "
          f"{min(cluster_sizes)}/{int(np.median(cluster_sizes))}/{max(cluster_sizes)} "
          f"| per-node (min/med/max) = "
          f"{int(sizes.min().item())}/{int(torch.median(sizes).item())}/{int(sizes.max().item())}")
   
    return DistributedData(
        train_input, train_output, wts, label_distribution, node_to_cluster
    )

# ----------------------------
# NEW: Pathological data distributor
# ----------------------------
def distribute_data_pathological(
    data_tuple,
    num_nodes: int,
    cluster_specs: List[List[int]], # List of label lists
    out_dim: int,
    device: torch.device,
    seed: int = 42
) -> DistributedData:
    """
    Distributes data based on explicit cluster-to-label specifications,
    partitioning data for overlapping labels and assigning nodes proportionally.
    
    This function is STRICT: it will fail with a ValueError if any
    configuration is invalid or results in a node receiving 0 data.
    """
    X, y = data_tuple
    N = len(X)
    assert N == len(y), "X and y length must match."
    
    num_clusters = len(cluster_specs)
    if num_clusters == 0:
        raise ValueError("[Data Dist:ERROR] cluster_specs cannot be empty.")

    # ---- NEW: Strict Input Check 1 ----
    if num_nodes < num_clusters:
        print(f"[Data Dist:ERROR] Bad Input: You have {num_nodes} nodes but {num_clusters} clusters.")
        print("  This is an invalid configuration. You must have at least as many nodes as clusters.")
        print("  Please increase --num_spokes or decrease the number of clusters in main.py and rerun.")
        raise ValueError("num_nodes must be >= num_clusters")

    # Seeding
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # ---- Per-class index pools
    class_indices = [[] for _ in range(out_dim)]
    for i, lbl in enumerate(y):
        class_indices[int(lbl.item())].append(i)
    
    # Shuffle each class index list once (for random slicing)
    for cls in range(out_dim):
        rng.shuffle(class_indices[cls])

    # ---- Logic for partitioning overlapping labels ----
    class_to_cluster_count = defaultdict(int)
    for spec in cluster_specs:
        for lbl in spec:
            if lbl < 0 or lbl >= out_dim: continue
            class_to_cluster_count[lbl] += 1
            
    class_slice_tracker = defaultdict(int) 
    label_chunks = {} 
    class_sizes_global = [len(ci) for ci in class_indices]
    
    for cid in range(num_clusters):
        for lbl in set(cluster_specs[cid]):
            if lbl < 0 or lbl >= out_dim: continue
            if (cid, lbl) in label_chunks: continue

            num_splits = class_to_cluster_count[lbl]
            if num_splits == 0: continue
                
            slice_idx = class_slice_tracker[lbl]
            class_slice_tracker[lbl] += 1
            
            total_size_for_class = class_sizes_global[lbl]
            
            if total_size_for_class == 0:
                label_chunks[(cid, lbl)] = (0, 0)
                continue
                
            num_per_cluster_slice = total_size_for_class // num_splits
            r_class = total_size_for_class % num_splits
            
            my_size = num_per_cluster_slice + (1 if slice_idx < r_class else 0)
            my_start = slice_idx * num_per_cluster_slice + min(slice_idx, r_class)
            my_end = my_start + my_size
            
            label_chunks[(cid, lbl)] = (my_start, my_end)

    # ---- Nodes per cluster (proportional to data size)
    
    # 1. Calculate total data size for each cluster spec (USING PARTITIONED SIZES)
    cluster_data_sizes = []
    for cid in range(num_clusters):
        size = 0
        for lbl in set(cluster_specs[cid]):
            if (cid, lbl) in label_chunks:
                start, end = label_chunks[(cid, lbl)]
                size += (end - start)
        cluster_data_sizes.append(size)

    # 2. Determine proportional node counts
    total_data_in_specs = sum(cluster_data_sizes)
    
    # ---- Strict Input Check 2 ----
    if total_data_in_specs == 0:
        print("[Data Dist:ERROR] Bad Input: No data was found for any of the provided cluster_specs.")
        print(f"  Specs: {cluster_specs}")
        print(f"  Data per cluster (partitioned): {cluster_data_sizes}")
        print("  This is an invalid configuration. Please check your specs (e.g., [[0,1], [2,3]]) and dataset, then rerun.")
        raise ValueError("No data found for any cluster spec. Aborting.")
    
    # Proportional split based on data size
    cluster_data_fracs = [size / total_data_in_specs for size in cluster_data_sizes]
    ideal_node_counts = [frac * num_nodes for frac in cluster_data_fracs]
    
    nodes_per_cluster_base = [int(np.floor(c)) for c in ideal_node_counts]
    remaining_nodes = num_nodes - sum(nodes_per_cluster_base)
    
    frac_parts = [c - np.floor(c) for c in ideal_node_counts]
    frac_order = np.argsort(frac_parts)[::-1]
    
    nodes_per_cluster = nodes_per_cluster_base[:]
    for i in range(remaining_nodes):
        cluster_to_add = frac_order[i % len(frac_order)]
        nodes_per_cluster[cluster_to_add] += 1
            
    print(f"[Data Dist] Cluster data sizes (partitioned): {cluster_data_sizes}")
    print(f"[Data Dist] Proportional node assignment: {nodes_per_cluster}")

    # 3. Create node->cluster mapping
    node_to_cluster = []
    for cid, cnt in enumerate(nodes_per_cluster):
        node_to_cluster.extend([cid] * cnt)
    node_base_offsets = np.cumsum([0] + nodes_per_cluster[:-1]).tolist()

    # ---- Initialize output lists
    train_input: List[torch.Tensor] = [None] * num_nodes
    train_output: List[torch.Tensor] = [None] * num_nodes

    # ---- Loop over clusters and assign data to nodes
    for cid in range(num_clusters):
        allowed_labels = sorted(list(set(cluster_specs[cid])))
        c_nodes = nodes_per_cluster[cid]
        
        if c_nodes == 0:
            # This is a valid outcome of proportional assignment (e.g., cluster has 0 data)
            # We just print a message and continue. The nodes assigned to this cluster
            # (zero) will not be processed, which is correct.
            print(f"[Data Dist] Info: Cluster {cid} was assigned 0 nodes due to proportional split.")
            continue
            
        base_node_idx = node_base_offsets[cid]

        # For each node in this cluster
        for j in range(c_nodes):
            node_idx = base_node_idx + j
            node_x_shards = []
            node_y_shards = []

            # For each class this cluster is allowed to see
            for lbl in allowed_labels:
                if (cid, lbl) not in label_chunks: continue
                
                chunk_start, chunk_end = label_chunks[(cid, lbl)]
                if chunk_start == chunk_end:
                    continue
                    
                cluster_class_indices = class_indices[lbl][chunk_start:chunk_end]
                total_size_for_class_in_this_cluster = len(cluster_class_indices)
                
                if total_size_for_class_in_this_cluster == 0:
                    continue

                # Calculate this node's stratified share
                num_per_node = total_size_for_class_in_this_cluster // c_nodes
                r_class = total_size_for_class_in_this_cluster % c_nodes
                
                my_size = num_per_node + (1 if j < r_class else 0)
                my_start = j * num_per_node + min(j, r_class)
                my_end = my_start + my_size
                
                if my_start >= my_end:
                    # This means more nodes than samples for this class-chunk
                    # We skip this class for this node, but the node may get
                    # data from other classes.
                    continue
                    
                my_indices = cluster_class_indices[my_start:my_end]
                
                if len(my_indices) > 0:
                    node_x_shards.append(X[my_indices])
                    node_y_shards.append(y[my_indices])

            # Combine all class shards for this node
            if len(node_x_shards) > 0:
                x_shard = torch.cat(node_x_shards, dim=0)
                y_shard = torch.cat(node_y_shards, dim=0)
                
                gen = torch.Generator()
                gen.manual_seed(seed + node_idx)
                perm = torch.randperm(len(x_shard), generator=gen)
                
                train_input[node_idx] = x_shard[perm].to(device)
                train_output[node_idx] = y_shard[perm].to(device)
            else:
                # ---- MODIFIED: Strict Input Check 3 ----
                # This node is assigned to the cluster, but received 0 total samples.
                # This happens if c_nodes > total_samples_in_cluster.
                # Per your request, this is a "bad input" error.
                print(f"[Data Dist:ERROR] Bad Input: Node {node_idx} (in Cluster {cid}) was assigned 0 data samples.")
                print(f"  This typically happens when a cluster has more nodes ({c_nodes}) than data samples ({cluster_data_sizes[cid]}).")
                print("  This is an invalid configuration for this experiment.")
                print("  Please reduce --num_spokes or adjust cluster specs so all nodes get data, then rerun.")
                raise ValueError(f"Node {node_idx} was assigned 0 data samples.")

    # ---- Final checks and wrap-up
    
    # ---- MODIFIED: Strict Input Check 4 (Bug Check) ----
    # This checks for `None`, which indicates a bug in *this function's* logic.
    none_nodes = [i for i, t in enumerate(train_input) if t is None]
    if none_nodes:
        print(f"[Data Dist:CRITICAL_BUG] Nodes {none_nodes} were not assigned any tensor.")
        print(f"  This indicates a bug in the data distribution logic (e.g., a cluster with c_nodes > 0 was skipped). Aborting.")
        raise RuntimeError("Data distribution logic failed to assign tensors to all nodes.")
    
    # ---- FedAvg weights and label histogram
    sizes = torch.tensor([len(t) for t in train_input], dtype=torch.float32, device=device)
    wts = sizes / (sizes.sum() + 1e-12) # Epsilon is fine, as total_data > 0 is guaranteed

    label_distribution = np.zeros((num_nodes, out_dim), dtype=int)
    for i in range(num_nodes):
        if len(train_output[i]) > 0:
            for yy in train_output[i].detach().cpu().tolist():
                if 0 <= yy < out_dim:
                    label_distribution[i][yy] += 1

    _print_node_size_stats([int(s.item()) for s in sizes])
    print(f"[Data Dist] Pathological split complete. node_to_cluster map: {node_to_cluster}")
    
    return DistributedData(
        train_input, train_output, wts, label_distribution, node_to_cluster
    )

def _print_node_size_stats(sizes_list):
    sizes = np.array(sizes_list, dtype=float)
    if sizes.size == 0:
        print("[Data Dist] No node sizes to report.")
        return
    mn, mx = int(np.min(sizes)), int(np.max(sizes))
    mean = float(np.mean(sizes))
    sd = float(np.std(sizes))
    print(f"[Data Dist] Per-node train samples => Min={mn}, Max={mx}, Mean={mean:.1f}, SD={sd:.1f}")

# ----------------------------
# Graph builders (only those used)
# ----------------------------
def create_k_random_regular_graph(n, k, device):
    W = torch.zeros((n,n), device=device)
    for i in range(n):
        W[i,i] = 1.0
        half = k // 2
        for j in range(1, half+1):
            W[i, (i+j) % n] = 1.0
            W[i, (i-j) % n] = 1.0
        if k % 2 == 1:
            W[i, (i+half+1) % n] = 1.0
    W = W / (W.sum(dim=1, keepdim=True))
    perm = torch.randperm(n, device=device)
    W = W[perm][:, perm]
    return W

def create_ring_graph(n, device):
    W = torch.zeros((n, n), device=device)
    for i in range(n):
        W[i, i] = 1/3
        W[i, (i-1) % n] = 1/3
        W[i, (i+1) % n] = 1/3
    return W

def create_torus_graph(n, device):
    s = int(math.sqrt(n))
    if s * s != n:
        raise ValueError("n must be a perfect square for torus.")
    W = torch.zeros((n, n), device=device)
    for i in range(s):
        for j in range(s):
            node = i * s + j
            nbrs = [
                ((i-1) % s) * s + j,
                ((i+1) % s) * s + j,
                i * s + (j-1) % s,
                i * s + (j+1) % s
            ]
            for v in nbrs:
                W[node, v] = 1/5
            W[node, node] = 1/5
    return W


# ----------------------------
# Model vec utils
# ----------------------------
def model_to_vec(net):
    return torch.cat([p.data.view(-1) for p in net.parameters()])

def vec_to_model(vec, net_name, num_inp, num_out, device):
    net = models.load_net(net_name, num_inp, num_out, device)
    idx = 0
    with torch.no_grad():
        for p in net.parameters():
            sz = p.numel()
            p.data = vec[idx:idx+sz].view(p.shape).clone().to(device)
            idx += sz
    return net


# ----------------------------
# Evaluation helpers
# ----------------------------
def evaluate_global_metrics(net, test_data, device):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, lab in test_data:
            x, lab = x.to(device), lab.to(device)
            out = net(x)
            loss = criterion(out, lab)
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)
    return total_loss, (correct / total) * 100.0

def get_input_shape(dataset_name):
    if dataset_name == 'mnist':         return [1, 28, 28]
    elif dataset_name == 'femnist':     return [1, 28, 28]
    elif dataset_name == 'cifar10':     return [3, 32, 32]
    elif dataset_name == 'agnews':      return [207]  # will be overwritten by actual max_len at trace-time if needed
    elif dataset_name == 'tiny_imagenet': return [3, 64, 64]
    elif dataset_name == 'cifar100':    return [3, 32, 32]
    else:                               raise ValueError(dataset_name)

def make_loader_from_tensors(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool=False):
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
