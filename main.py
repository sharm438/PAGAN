"""
main_v2.py
==========
Extends main_0305.py with the PAGANv2 protocol.
All original paths (fedavg, fedsgd, isolated, pagan, dpfl, k-regular) are
preserved identically.  PAGANv2 is added under --topo paganv2.

New args (all prefixed v2_):
  --v2_warmup_rounds    int     default 20
  --v2_tau_0            float   default 2.0
  --v2_tau_min          float   default 0.2
  --v2_tau_half_life    float   default 200.0
  --v2_tau_fixed        float   default None  (overrides schedule)
  --v2_shadow_window    int     default 20
  --v2_softmax_tau      float   default 1.0
  --v2_freeze_affinity  flag    freeze affinity at warmup end
  --v2_freeze_embeddings flag   freeze embeddings at warmup end
  --v2_emb_lr           float   default 0.1  (lr for v2 embeddings)
  --v2_emb_lr_post      float   default 0.01 (lr after warmup if not frozen)
  --v2_emb_steps        int     default 10
  --v2_eff_thresh       float   default 0.02 (soft TP/FP threshold)

Experiment ladder:
  Exp 1 (broad baseline):
    --topo paganv2 --v2_freeze_affinity --v2_freeze_embeddings --v2_tau_fixed 2.0
  Exp 2 (tight baseline):
    --topo paganv2 --v2_freeze_affinity --v2_freeze_embeddings --v2_tau_fixed 0.2
  Exp 3 (live affinity + schedule):
    --topo paganv2 --v2_freeze_embeddings
  Exp 4 (full v2):
    --topo paganv2
"""

import argparse
import torch
import os
import json
from copy import deepcopy
import utils as utils
import aggregation as agg
import train_node as train_node
import models as models
import math

import torch.multiprocessing as mp
import eval_worker as eval_worker

from functools import partial
from multiprocessing import Pool, get_start_method

import pdb
import numpy as np
import random

from sklearn.model_selection import train_test_split

# PAGANv2
from shadow_registry import ShadowRegistry
from pagan_v2 import PAGANv2
# PAGANv3
from pagan_v3 import PAGANv3
from pagan_v4 import PAGANv4

from fedspd import FedSPD
from l2c import L2CProtocol
from federico import FedeRiCoProtocol


# -----------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # ── original args (unchanged) ──────────────────────────────────────
    parser.add_argument("--exp",            type=str,   default='experiment')
    parser.add_argument("--dataset",        type=str,   default='cifar10',
                        choices=['mnist','cifar10','femnist','agnews',
                                 'tiny_imagenet','cifar100'])
    parser.add_argument("--fraction",       type=float, default=1.0)
    parser.add_argument("--bias",           type=float, default=0.1)
    parser.add_argument("--aggregation",    type=str,   default='p2p',
                        choices=['isolated','fedsgd','fedavg','p2p'])
    parser.add_argument("--topo",           type=str,   default=None,
                        choices=['k-regular-random','k-regular-clustered',
                                 'pagan','dpfl','paganv2','paganv3', 'paganv4', 'fedspd', 'l2c', 'federico'])
    parser.add_argument("--num_spokes",     type=int,   default=100)
    parser.add_argument("--num_clusters",   type=int,   default=0)
    parser.add_argument("--num_rounds",     type=int,   default=500)
    parser.add_argument("--num_local_iters",type=int,   default=5)
    parser.add_argument("--batch_size",     type=int,   default=None)
    parser.add_argument("--eval_time",      type=int,   default=10)
    parser.add_argument("--gpu",            type=int,   default=0)
    parser.add_argument("--k",              type=int,   default=20)

    parser.add_argument("--lr",             type=float, default=0.1)
    parser.add_argument("--sample_type",    type=str,   default="round_robin",
                        choices=["round_robin","random"])
    parser.add_argument("--seed",           type=int,   default=108)
    parser.add_argument("--dist_case",      type=str,   default='dirichlet',
                        choices=['dirichlet','patho_1','patho_2','patho_3'])
    parser.add_argument("--local_test_frac",type=float, default=0.2)
    parser.add_argument("--eval_local",     action="store_true")
    parser.add_argument("--num_workers",    type=int,   default=10)
    parser.add_argument("--embed_dim",      type=int,   default=8)

    parser.add_argument("--val_frac",       type=float, default=0.0)
    parser.add_argument("--use_external_test", action="store_true",
                        help="Distribute CIFAR-10 test set with same proportions as train")
    # ── PAGANv2-specific args ──────────────────────────────────────────
    parser.add_argument("--v2_warmup_rounds",   type=int,   default=25)
    parser.add_argument("--v2_ema_lambda",      type=float, default=0.95,
                        help="EMA decay for distance history (0.9=fast, 0.99=slow).")
    parser.add_argument("--v2_tau_0",           type=float, default=2.0,
                        help="Starting aggregation temperature post-warmup.")
    parser.add_argument("--v2_tau_min",         type=float, default=0.5,
                        help="Minimum aggregation temperature.")
    parser.add_argument("--v2_tau_half_life",   type=float, default=200.0,
                        help="Rounds post-warmup for tau to reach midpoint.")
    parser.add_argument("--v2_softmax_tau",     type=float, default=1.0,
                        help="Temperature for diagnostic shadow_sw recording.")
    parser.add_argument("--v2_shadow_window",   type=int,   default=20)
    parser.add_argument("--v2_freeze_ema",      action="store_true",
                        help="Freeze EMA distances at warmup end (ablation).")
    parser.add_argument("--v2_freeze_embeddings", action="store_true",
                        help="Freeze embedding updates at warmup end (ablation).")
    parser.add_argument("--v2_emb_lr",         type=float, default=0.1)
    parser.add_argument("--v2_emb_lr_post",    type=float, default=0.01)
    parser.add_argument("--v2_emb_steps",      type=int,   default=10)
    parser.add_argument("--v2_eff_thresh",      type=float, default=0.02)

    # v3 reuses all v2_* args (warmup, ema_lambda, tau_*, emb_*).
    # Only the blend schedule and triplet scheme are new.
    parser.add_argument("--v3_alpha_start",    type=int,   default=None,
                        help="Round where α begins decaying from 1.0. "
                             "Default: v2_warmup_rounds.")
    parser.add_argument("--v3_alpha_end",      type=int,   default=None,
                        help="Round where α reaches 0.0 (pure embedding). "
                             "Default: num_rounds.")
    parser.add_argument("--v3_triplet_scheme", type=str,   default='flat',
                        choices=['flat', 'inv_rank', 'inv_sqrt_rank'],
                        help="Triplet loss rank-weighting scheme. "
                             "'flat'=v2 baseline, 'inv_rank'=1/(r+1), "
                             "'inv_sqrt_rank'=1/sqrt(r+1).")
    parser.add_argument("--v3_alpha_schedule", type=str, default='linear',
                    choices=['linear','cosine','concave','convex','floor'])



    # ── FedSPD-specific args ──────────────────────────────────────────
    parser.add_argument("--fedspd_S",    type=int,   default=2,
                        help="Number of cluster centers per node.")
    parser.add_argument("--fedspd_ckc",  type=int,   default=10,
                        help="Label-switch alignment every N rounds.")
    parser.add_argument("--fedspd_init", type=str,   default='fedspd_style',
                        choices=['fedspd_style', 'round_robin'],
                        help="Cluster center + data initialization strategy.")
    # ── L2C-specific args ─────────────────────────────────────────────
    parser.add_argument("--l2c_beta",    type=float, default=0.01,
                        help="Learning rate for L2C mixing weight logits.")
    parser.add_argument("--l2c_warmup",  type=int,   default=25,
                        help="Rounds of random sampling before topology freeze.")
    # ── FedeRiCo-specific args ────────────────────────────────────────
    parser.add_argument("--federico_epsilon", type=float, default=0.3,
                        help="Exploration probability in ε-greedy sampling.")
    parser.add_argument("--federico_beta",    type=float, default=0.6,
                        help="EMA momentum for loss tracking.")

    # ── PAGANv4-specific args ──────────────────────────────────────────
    # v4 reuses all v2_* args for EMA, warmup, tau schedule.
    # New knobs are only the trust-gate parameters.
    parser.add_argument("--v4_n_buckets",          type=int,   default=4,
                        help="Rank buckets for warmup bucket tracking.")
    parser.add_argument("--v4_friend_thresh",      type=float, default=0.3,
                        help="W_warmup above this → warmup friend (used in vouching).")
    parser.add_argument("--v4_conflict_bucket",    type=int,   default=2,
                        help="warmup_bucket >= this with 3+ obs → conflict pair.")
    parser.add_argument("--v4_conflict_cap",       type=float, default=0.15,
                        help="Max trust for conflict pairs regardless of vouching.")
    parser.add_argument("--v4_vouch_M",            type=int,   default=5,
                        help="Top-M of j's neighbours checked for vouching.")
    parser.add_argument("--v4_trust_log_eps",      type=float, default=1e-3,
                        help="Epsilon in log(trust+eps); floor for logits.")


    return parser.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(args):
    if args.num_workers > 0:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    if args.seed > 0:
        random.seed(args.seed);  np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    aggregator_device = torch.device(
        f'cuda:{args.gpu}' if (args.gpu >= 0 and torch.cuda.is_available()) else 'cpu')

    outputs_path = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_path, exist_ok=True)
    filename = os.path.join(outputs_path, args.exp)

    # ── Data loading (identical to main_0305) ─────────────────────────
    local_test_pairs = None
    distributed_val_data = distributed_val_label = None

    if args.dataset == 'femnist':
        trainObject, distributed_data, distributed_label = utils.load_data(
            args.dataset, 32, args.lr, args.fraction)
        test_data   = trainObject.test_data
        inp_dim     = trainObject.num_inputs
        out_dim     = trainObject.num_outputs
        lr          = args.lr
        batch_size  = trainObject.batch_size
        net_name    = trainObject.net_name
        num_clients = len(distributed_label)
        counts      = torch.tensor([distributed_label[i].shape[0]
                                     for i in range(num_clients)], dtype=torch.float32)
        node_weights = (counts / counts.sum()).to(aggregator_device)
    else:
        trainObject = utils.load_data(args.dataset, args.batch_size, args.lr, args.fraction)
        data, labels = trainObject.train_data, trainObject.train_labels
        test_data    = trainObject.test_data      # global DataLoader (unchanged, for global eval)
        inp_dim      = trainObject.num_inputs
        out_dim      = trainObject.num_outputs
        lr           = args.lr
        batch_size   = trainObject.batch_size
        net_name     = trainObject.net_name

        current_seed = args.seed if args.seed > 0 else 42
        if args.dist_case == 'dirichlet':
            distributedData = utils.clustered_distribute_data(
                (data, labels), num_nodes=args.num_spokes,
                num_clusters=(args.num_spokes if args.num_clusters == 0
                              else args.num_clusters),
                alpha=args.bias, out_dim=out_dim,
                device=torch.device("cpu"), seed=42)
        else:
            cluster_specs = {
                'patho_1': [[0,1,2,3,4],[5,6,7,8,9]],
                'patho_2': [[0,1,2,3],[2,3,4,5],[6,7],[8,9],[0,8]],
                'patho_3': [[0,1,2,3],[5,6,7,8],[4,9]],
            }[args.dist_case]
            distributedData = utils.distribute_data_pathological(
                (data, labels), num_nodes=args.num_spokes,
                cluster_specs=cluster_specs, out_dim=out_dim,
                device=torch.device("cpu"), seed=current_seed)

        distributed_data  = distributedData.train_input
        distributed_label = distributedData.train_output
        node_weights      = distributedData.wts.to(aggregator_device)
        node_to_cluster   = np.array(distributedData.node_to_cluster)
        jsd_matrix        = utils.compute_pairwise_jsd(
            distributedData.label_distribution).to(aggregator_device)

    # ── Data splits ───────────────────────────────────────────────────
    if args.use_external_test:
        # Extract raw tensors from global test DataLoader
        _bx, _by = [], []
        for bx, by in trainObject.test_data:
            _bx.append(bx); _by.append(by)
        raw_test_data   = torch.cat(_bx, dim=0)
        raw_test_labels = torch.cat(_by, dim=0)

        # Train+Val only (no local test carved from train set)
        splits = utils._make_local_splits(
            distributed_data, distributed_label,
            0.0, args.val_frac, args.seed, print_stats=False)

        distributed_train_data  = splits["train_data"]
        distributed_train_label = splits["train_label"]

        if args.val_frac > 0:
            distributed_val_data  = splits["val_data"]
            distributed_val_label = splits["val_label"]

        # Distribute external test set with same per-node class proportions
        ext_test_data, ext_test_label = utils.distribute_test_data(
            raw_test_data, raw_test_labels,
            distributedData.cluster_class_counts,
            distributedData.node_to_cluster,
            out_dim, torch.device("cpu"), seed=42)
        local_test_pairs = list(zip(ext_test_data, ext_test_label))

        split_distribution_stats = utils.analyze_train_test_split_distribution(
            distributed_train_label, ext_test_label, out_dim)
    else:
        # Original path: carve local test (and optionally val) from train set
        splits = utils._make_local_splits(
            distributed_data, distributed_label,
            args.local_test_frac, args.val_frac, args.seed, print_stats=False)

        distributed_train_data  = splits["train_data"]
        distributed_train_label = splits["train_label"]

        if args.local_test_frac > 0:
            split_distribution_stats = utils.analyze_train_test_split_distribution(
                splits["train_label"], splits["test_label"], out_dim)
            local_test_pairs = list(zip(splits["test_data"], splits["test_label"]))
        if args.val_frac > 0:
            distributed_val_data  = splits["val_data"]
            distributed_val_label = splits["val_label"]

    if args.eval_local and local_test_pairs is None:
        print("[Info] --eval_local given but no local test data available.")

    # if args.dataset == 'femnist':
    #     trainObject, distributed_data, distributed_label = utils.load_data(
    #         args.dataset, 32, args.lr, args.fraction)
    #     test_data   = trainObject.test_data
    #     inp_dim     = trainObject.num_inputs
    #     out_dim     = trainObject.num_outputs
    #     lr          = args.lr
    #     batch_size  = trainObject.batch_size
    #     net_name    = trainObject.net_name
    #     num_clients = len(distributed_label)
    #     counts      = torch.tensor([distributed_label[i].shape[0]
    #                                  for i in range(num_clients)], dtype=torch.float32)
    #     node_weights = (counts / counts.sum()).to(aggregator_device)
    # else:
    #     trainObject = utils.load_data(args.dataset, args.batch_size, args.lr, args.fraction)
    #     data, labels = trainObject.train_data, trainObject.train_labels
    #     test_data    = trainObject.test_data
    #     inp_dim      = trainObject.num_inputs
    #     out_dim      = trainObject.num_outputs
    #     lr           = args.lr
    #     batch_size   = trainObject.batch_size
    #     net_name     = trainObject.net_name

    #     current_seed = args.seed if args.seed > 0 else 42
    #     if args.dist_case == 'dirichlet':
    #         distributedData = utils.clustered_distribute_data(
    #             (data, labels), num_nodes=args.num_spokes,
    #             num_clusters=(args.num_spokes if args.num_clusters == 0
    #                           else args.num_clusters),
    #             alpha=args.bias, out_dim=out_dim,
    #             device=torch.device("cpu"), seed=42)
    #     else:
    #         cluster_specs = {
    #             'patho_1': [[0,1,2,3,4],[5,6,7,8,9]],
    #             'patho_2': [[0,1,2,3],[2,3,4,5],[6,7],[8,9],[0,8]],
    #             'patho_3': [[0,1,2,3],[5,6,7,8],[4,9]],
    #         }[args.dist_case]
    #         distributedData = utils.distribute_data_pathological(
    #             (data, labels), num_nodes=args.num_spokes,
    #             cluster_specs=cluster_specs, out_dim=out_dim,
    #             device=torch.device("cpu"), seed=current_seed)

    #     distributed_data  = distributedData.train_input
    #     distributed_label = distributedData.train_output
    #     node_weights      = distributedData.wts.to(aggregator_device)
    #     node_to_cluster   = np.array(distributedData.node_to_cluster)
    #     jsd_matrix        = utils.compute_pairwise_jsd(
    #         distributedData.label_distribution).to(aggregator_device)

    # splits = utils._make_local_splits(
    #     distributed_data, distributed_label,
    #     args.local_test_frac, args.val_frac, args.seed, print_stats=False)

    # if args.local_test_frac > 0:
    #     split_distribution_stats = utils.analyze_train_test_split_distribution(
    #         splits["train_label"], splits["test_label"], out_dim)

    # distributed_train_data  = splits["train_data"]
    # distributed_train_label = splits["train_label"]

    # if args.local_test_frac > 0:
    #     local_test_pairs = list(zip(splits["test_data"], splits["test_label"]))
    # if args.val_frac > 0:
    #     distributed_val_data  = splits["val_data"]
    #     distributed_val_label = splits["val_label"]

    # if args.eval_local and local_test_pairs is None:
    #     print("[Info] --eval_local given but local_test_frac==0.")

    # ── Global model ──────────────────────────────────────────────────
    global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)
    rr_indices   = {i: 0 for i in range(args.num_spokes)}

    # ── Protocol initialization ───────────────────────────────────────
    prev_ranked_neighbors = None
    prev_ranked_dists     = None
    E_list = embedding_opts = embedding_schedulers = None
    pagan_protocol   = None
    pagan_v2_protocol = None

    # ── Original PAGAN init ───────────────────────────────────────────
    if args.aggregation == 'p2p' and args.topo == 'pagan':
        N = args.num_spokes
        g = torch.Generator(device=aggregator_device)
        g.manual_seed(args.seed if args.seed and args.seed > 0 else 123456789)
        E_list = agg.init_identical_E_list(N, args.embed_dim, aggregator_device, g)
        embedding_opts = []
        embedding_schedulers = []
        for i in range(args.num_spokes):
            opt = torch.optim.SGD([E_list[i]], lr=0.1, momentum=0.9, weight_decay=1e-4)
            embedding_opts.append(opt)
            embedding_schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99))
        pagan_protocol = agg.PAGANProtocol(
            args.num_spokes, aggregator_device, node_to_cluster=node_to_cluster)
        current_eps = args.eps

    # ── PAGANv2 init ──────────────────────────────────────────────────
    if args.aggregation == 'p2p' and args.topo == 'paganv2':
        N = args.num_spokes
        g = torch.Generator(device=aggregator_device)
        g.manual_seed(args.seed if args.seed and args.seed > 0 else 123456789)
        E_list = agg.init_identical_E_list(N, args.embed_dim, aggregator_device, g)

        embedding_opts = []
        embedding_schedulers = []
        for i in range(N):
            opt = torch.optim.SGD([E_list[i]], lr=args.v2_emb_lr,
                                   momentum=0.9, weight_decay=1e-4)
            embedding_opts.append(opt)
            embedding_schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99))

        pagan_v2_protocol = PAGANv2(
            num_nodes         = N,
            device            = aggregator_device,
            total_rounds      = args.num_rounds,
            warmup_rounds     = args.v2_warmup_rounds,
            ema_lambda        = args.v2_ema_lambda,
            tau_0             = args.v2_tau_0,
            tau_min           = args.v2_tau_min,
            tau_half_life     = args.v2_tau_half_life,
            softmax_tau       = args.v2_softmax_tau,
            shadow_window     = args.v2_shadow_window,
            freeze_ema        = args.v2_freeze_ema,
            freeze_embeddings = args.v2_freeze_embeddings,
            node_to_cluster   = node_to_cluster,
            eff_weight_thresh = args.v2_eff_thresh,
            debug_node        = 0,
        )

        print(f"[PAGANv2] warmup={args.v2_warmup_rounds}  "
              f"ema_λ={args.v2_ema_lambda}  "
              f"τ: {args.v2_tau_0}→{args.v2_tau_min} (half-life={args.v2_tau_half_life})  "
              f"freeze_ema={args.v2_freeze_ema}  "
              f"freeze_emb={args.v2_freeze_embeddings}")

    # ── PAGANv3 init ──────────────────────────────────────────────────
    pagan_v3_protocol = None
    if args.aggregation == 'p2p' and args.topo == 'paganv3':
        N = args.num_spokes
        g = torch.Generator(device=aggregator_device)
        g.manual_seed(args.seed if args.seed and args.seed > 0 else 123456789)
        E_list = agg.init_identical_E_list(N, args.embed_dim,
                                           aggregator_device, g)
        embedding_opts = []
        embedding_schedulers = []
        for i in range(N):
            opt = torch.optim.SGD([E_list[i]], lr=args.v2_emb_lr,
                                   momentum=0.9, weight_decay=1e-4)
            embedding_opts.append(opt)
            embedding_schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99))
 
        pagan_v3_protocol = PAGANv3(
            num_nodes            = N,
            device               = aggregator_device,
            total_rounds         = args.num_rounds,
            warmup_rounds        = args.v2_warmup_rounds,
            ema_lambda           = args.v2_ema_lambda,
            tau_0                = args.v2_tau_0,
            tau_min              = args.v2_tau_min,
            tau_half_life        = args.v2_tau_half_life,
            softmax_tau          = args.v2_softmax_tau,
            shadow_window        = args.v2_shadow_window,
            freeze_ema           = args.v2_freeze_ema,
            freeze_embeddings    = args.v2_freeze_embeddings,
            node_to_cluster      = node_to_cluster,
            eff_weight_thresh    = args.v2_eff_thresh,
            debug_node           = 0,
            alpha_start          = args.v3_alpha_start,
            alpha_end            = args.v3_alpha_end,
            triplet_weight_scheme= args.v3_triplet_scheme,
        )
        # metrics['v3_soft_metrics'] = []
        # metrics['v3_emb_stats']    = []

    
    global_wts  = utils.model_to_vec(global_model)
    # ── FedSPD init ─────────
    fedspd_protocol = None
    if args.aggregation == 'p2p' and args.topo == 'fedspd':
        D = global_wts.shape[0]
        fedspd_protocol = FedSPD(
            N=args.num_spokes,
            S=args.fedspd_S,
            D=D,
            net_name=net_name,
            inp_dim=inp_dim,
            out_dim=out_dim,
            device=aggregator_device,
            k=args.k,
            seed=args.seed,
            train_data=distributed_train_data,
            train_labels=distributed_train_label,
            lr=args.lr,
            batch_size=batch_size,
            num_local_iters=args.num_local_iters,
            init_style=args.fedspd_init,
            ckc=args.fedspd_ckc,
        )

    # ── L2C init ──────────────────────────────────────────────────────
    l2c_protocol = None
    if args.aggregation == 'p2p' and args.topo == 'l2c':
        if distributed_val_data is None or all(len(v) == 0 for v in distributed_val_data):
            raise ValueError("[L2C] Requires --val_frac > 0 to provide local validation sets.")
        l2c_protocol = L2CProtocol(
            N=args.num_spokes,
            k=args.k,
            warmup_rounds=args.l2c_warmup,
            beta=args.l2c_beta,
            device=aggregator_device,
            seed=args.seed,
        )
        print(f"[L2C] N={args.num_spokes}  k={args.k}  "
              f"warmup={args.l2c_warmup}  beta={args.l2c_beta}")

    # ── FedeRiCo init ─────────────────────────────────────────────────
    federico_protocol = None
    if args.aggregation == 'p2p' and args.topo == 'federico':
        federico_protocol = FedeRiCoProtocol(
            N=args.num_spokes,
            M=args.k,
            epsilon=args.federico_epsilon,
            beta=args.federico_beta,
            device=aggregator_device,
            seed=args.seed,
        )
        print(f"[FedeRiCo] N={args.num_spokes}  M={args.k}  "
              f"ε={args.federico_epsilon}  β={args.federico_beta}")

    # ── PAGANv4 init ──────────────────────────────────────────────────
    pagan_v4_protocol = None
    if args.aggregation == 'p2p' and args.topo == 'paganv4':
        N = args.num_spokes
        g = torch.Generator(device=aggregator_device)
        g.manual_seed(args.seed if args.seed and args.seed > 0 else 123456789)
        E_list = agg.init_identical_E_list(N, args.embed_dim, aggregator_device, g)
 
        embedding_opts = []
        embedding_schedulers = []
        for i in range(N):
            opt = torch.optim.SGD([E_list[i]], lr=args.v2_emb_lr,
                                   momentum=0.9, weight_decay=1e-4)
            embedding_opts.append(opt)
            embedding_schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99))
 
        pagan_v4_protocol = PAGANv4(
            num_nodes                = N,
            device                   = aggregator_device,
            total_rounds             = args.num_rounds,
            warmup_rounds            = args.v2_warmup_rounds,
            ema_lambda               = args.v2_ema_lambda,
            tau_0                    = args.v2_tau_0,
            tau_min                  = args.v2_tau_min,
            tau_half_life            = args.v2_tau_half_life,
            softmax_tau              = args.v2_softmax_tau,
            shadow_window            = args.v2_shadow_window,
            k_sample                 = args.k,
            freeze_ema               = args.v2_freeze_ema,
            freeze_embeddings        = args.v2_freeze_embeddings,
            node_to_cluster          = node_to_cluster,
            eff_weight_thresh        = args.v2_eff_thresh,
            debug_node               = 0,
            # v4-specific
            n_buckets                = args.v4_n_buckets,
            warmup_friend_threshold  = args.v4_friend_thresh,
            conflict_bucket_threshold= args.v4_conflict_bucket,
            conflict_cap             = args.v4_conflict_cap,
            vouch_top_M              = args.v4_vouch_M,
            trust_log_eps            = args.v4_trust_log_eps,
        )
 
        print(f"[PAGANv4] warmup={args.v2_warmup_rounds}  "
              f"λ={args.v2_ema_lambda}  "
              f"τ: {args.v2_tau_0}→{args.v2_tau_min} (hl={args.v2_tau_half_life})  "
              f"friend_thresh={args.v4_friend_thresh}  "
              f"conflict_bucket≥{args.v4_conflict_bucket}  "
              f"conflict_cap={args.v4_conflict_cap}  "
              f"vouch_M={args.v4_vouch_M}")
 


    # ── DPFL init ─────────────────────────────────────────────────────
    if args.aggregation == 'p2p' and args.topo == 'dpfl':
        dpfl_val_map  = {}
        dpfl_selector = agg.DPFLSelector(args.num_spokes, aggregator_device)
        total_val_samples = sum([len(v) for v in distributed_val_data])
        if total_val_samples == 0:
            print("[WARNING] DPFL: Val set is empty!")
        for i in range(args.num_spokes):
            dpfl_val_map[i] = (distributed_val_data[i], distributed_val_label[i])

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = {
        'round': [], 'global_acc': [], 'global_loss': [],
        'local_acc': [], 'local_loss': [],
        'node_to_cluster': [], 'purity_exploit': [],
        'avg_scope': [], 'phase': [],
        'fp_knee_strengths': [], 'fn_knee_strengths': [],
        'tp': [], 'fp': [],
        'ratio_fn_dist': [], 'ratio_tn_dist': [],
        'ratio_tp_dist': [], 'ratio_fp_dist': [],
        # v2-specific
        'v2_soft_metrics': [],
        'v2_spearman': [],
        'v2_emb_stats': [],
        # v3-specific
        'v3_soft_metrics': [],
        'v3_emb_stats': [],
        'v4_soft_metrics': [],
        'v4_emb_stats': [],
        'v4_trust_snapshots': [],
    }
    if args.local_test_frac > 0:
        metrics['local_split_stats'] = split_distribution_stats
    if args.num_clusters > 0:
        metrics['node_to_cluster'] = node_to_cluster.tolist()
    if args.topo == 'paganv2':
        metrics['v2_warmup_rounds'] = args.v2_warmup_rounds
    # if args.topo == 'paganv3':
    #     metrics['v3_warmup_rounds'] = args.v2_warmup_rounds
    #     metrics['v3_emb_mature_round'] = args.v3_emb_mature_round
    if args.topo == 'paganv4':
        metrics['v4_warmup_rounds'] = args.v2_warmup_rounds

    # ── Training state ────────────────────────────────────────────────
    is_fed_path = args.aggregation in ('fedavg', 'fedsgd')
    local_iters_effective = 1 if args.aggregation == 'fedsgd' else args.num_local_iters
    
    node_states = global_wts.detach().unsqueeze(0).repeat(
        args.num_spokes, 1).to(aggregator_device)

    worker_pool = None
    if args.aggregation == 'p2p' and args.num_workers > 0:
        worker_pool = mp.Pool(
            processes=args.num_workers,
            initializer=eval_worker.init_worker,
            initargs=(args.dataset, net_name, inp_dim, out_dim,
                      batch_size, args.gpu, args.fraction))

    # ================================================================
    # Training loop
    # ================================================================
    try:
        for rnd in range(args.num_rounds):

            # # ── Local training ────────────────────────────────────────
            # if args.aggregation in ('fedavg', 'fedsgd'):
            #     start_vec   = global_wts.detach()
            #     node_states = start_vec.unsqueeze(0).repeat(args.num_spokes, 1)

            # velocities = []
            # for node_id in range(args.num_spokes):
            #     start_vec = None
            #     if args.aggregation in ('isolated', 'p2p'):
            #         start_vec = node_states[node_id].detach().clone()

            #     updated_wts = train_node.local_train_worker_inline(
            #         node_id, start_vec,
            #         distributed_train_data[node_id],
            #         distributed_train_label[node_id],
            #         inp_dim, out_dim, net_name,
            #         local_iters_effective, batch_size, args.lr,
            #         aggregator_device, args.sample_type, rr_indices)

            #     if args.topo in ('pagan', 'paganv2', 'paganv3') and start_vec is not None:
            #         velocities.append(
            #             torch.norm(updated_wts - start_vec).item())

            #     node_states[node_id] = updated_wts.detach()

            # ── Local training ────────────────────────────────────────
            if args.aggregation in ('fedavg', 'fedsgd'):
                start_vec   = global_wts.detach()
                node_states = start_vec.unsqueeze(0).repeat(args.num_spokes, 1)

            # ── FedSPD: train + aggregate + recluster in one call ─────
            if args.aggregation == 'p2p' and args.topo == 'fedspd':
                node_states = fedspd_protocol.run_round(
                    rnd, distributed_train_data, distributed_train_label)
            else:
                velocities = []
                for node_id in range(args.num_spokes):
                    start_vec = None
                    if args.aggregation in ('isolated', 'p2p'):
                        start_vec = node_states[node_id].detach().clone()

                    updated_wts = train_node.local_train_worker_inline(
                        node_id, start_vec,
                        distributed_train_data[node_id],
                        distributed_train_label[node_id],
                        inp_dim, out_dim, net_name,
                        local_iters_effective, batch_size, args.lr,
                        aggregator_device, args.sample_type, rr_indices)

                    if args.topo in ('pagan', 'paganv2', 'paganv3') and start_vec is not None:
                        velocities.append(
                            torch.norm(updated_wts - start_vec).item())

                    node_states[node_id] = updated_wts.detach()

            # ── Aggregation ───────────────────────────────────────────
            if args.aggregation == 'isolated':
                pass

            elif is_fed_path:
                global_wts = agg.federated_aggregation(
                    node_states, node_weights).detach()
                global_model = utils.vec_to_model(
                    global_wts.to(aggregator_device),
                    net_name, inp_dim, out_dim, aggregator_device)
                utils.evaluate_and_log(
                    current_round=rnd + 1, metrics=metrics,
                    model_source=global_model, mode='federated',
                    test_data=test_data, local_test_pairs=local_test_pairs,
                    device=aggregator_device, batch_size=batch_size,
                    dataset_name=args.dataset,
                    net_name=net_name, inp_dim=inp_dim, out_dim=out_dim,
                    eval_local=args.eval_local)

            # ── Original PAGAN ────────────────────────────────────────
            elif args.aggregation == 'p2p' and args.topo == 'pagan':
                with torch.no_grad():
                    neigh_lists = []
                    for i in range(args.num_spokes):
                        phys_list = (prev_ranked_neighbors[i].tolist()
                                     if prev_ranked_neighbors is not None else [])
                        targets_i = pagan_protocol.get_sampling_targets(
                            rnd, i, args.k, phys_list, E_list[i])
                        neigh_lists.append(torch.tensor(
                            targets_i, device=aggregator_device, dtype=torch.long))

                    ranked_neighbors, ranked_dists = \
                        agg.rank_neighbors_by_model_distance(node_states, neigh_lists)

                    r0_tp = sum(
                        1 for idx in range(args.num_spokes)
                        if ranked_neighbors[idx].numel() > 0 and
                        node_to_cluster[ranked_neighbors[idx][0].item()] ==
                        node_to_cluster[idx])
                    print(f"Round {rnd} [PAGAN] Rank-0 TP={r0_tp}/{args.num_spokes}")

                    new_states = pagan_protocol.run_topology_and_aggregate(
                        rnd, node_states, E_list, ranked_neighbors,
                        prev_ranked_neighbors)
                    node_states.copy_(new_states)

                if prev_ranked_neighbors is None:
                    src_ranks = ranked_neighbors
                    src_dists = ranked_dists
                    selection = [torch.tensor([i], device=aggregator_device,
                                              dtype=torch.long)
                                 for i in range(args.num_spokes)]
                else:
                    src_ranks = prev_ranked_neighbors
                    src_dists = prev_ranked_dists
                    selection = [torch.cat([ranked_neighbors[i],
                                  torch.tensor([i], device=aggregator_device,
                                               dtype=torch.long)])
                                 for i in range(args.num_spokes)]

                neighbor_ranklists = agg.gather_neighbor_ranklists_with_dists(
                    src_ranks, src_dists, chosen_neighbors=selection)
                if rnd >= 1:
                    agg.update_embeddings_ladder_triplet(
                        E_list, neighbor_ranklists,
                        opt_list=embedding_opts, steps=10)
                for sch in embedding_schedulers:
                    sch.step()
                prev_ranked_neighbors = ranked_neighbors
                prev_ranked_dists     = ranked_dists

            # ================================================================
            # PAGANv2
            # ================================================================
            elif args.aggregation == 'p2p' and args.topo == 'paganv2':
                with torch.no_grad():
                    # 1. Build sampling targets
                    neigh_lists = []
                    for i in range(args.num_spokes):
                        phys_list = (prev_ranked_neighbors[i].tolist()
                                     if prev_ranked_neighbors is not None else [])
                        targets_i = pagan_v2_protocol.get_sampling_targets(
                            rnd, i, args.k, phys_list, E_list[i])
                        neigh_lists.append(torch.tensor(
                            targets_i, device=aggregator_device, dtype=torch.long))

                    # 2. Rank by physical model distance
                    ranked_neighbors, ranked_dists = \
                        agg.rank_neighbors_by_model_distance(node_states, neigh_lists)

                    # 3. Record into shadow registry
                    pagan_v2_protocol.record(rnd, ranked_neighbors, ranked_dists)

                    # 4. Rank-0 diagnostic
                    # r0_tp = sum(
                    #     1 for idx in range(args.num_spokes)
                    #     if ranked_neighbors[idx].numel() > 0 and
                    #     node_to_cluster[ranked_neighbors[idx][0].item()] ==
                    #     node_to_cluster[idx])
                    # print(f"Round {rnd:3d} [v2]"
                    #       f"  R0-TP={r0_tp}/{args.num_spokes}")

                    # 5. Aggregate (warmup = isolation, post-warmup = weighted)
                    new_states = pagan_v2_protocol.run_topology_and_aggregate(
                        rnd, node_states, E_list, ranked_neighbors,
                        prev_ranked_neighbors)
                    node_states.copy_(new_states)

                # 6. Embedding update
                emb_frozen = (args.v2_freeze_embeddings and
                              rnd >= args.v2_warmup_rounds)
                if not emb_frozen and prev_ranked_neighbors is not None:
                    # Social triangulation evidence (same as PAGAN)
                    selection = [
                        torch.cat([ranked_neighbors[i],
                                   torch.tensor([i], device=aggregator_device,
                                                dtype=torch.long)])
                        for i in range(args.num_spokes)
                    ]
                    evidence = agg.gather_neighbor_ranklists_with_dists(
                        prev_ranked_neighbors, prev_ranked_dists,
                        chosen_neighbors=selection)

                    # Halve steps post-warmup
                    steps = (args.v2_emb_steps if rnd < args.v2_warmup_rounds
                             else max(1, args.v2_emb_steps // 2))

                    # Switch to lower lr post-warmup (if not frozen)
                    if rnd == args.v2_warmup_rounds and not args.v2_freeze_embeddings:
                        embedding_schedulers = []
                        for opt in embedding_opts:
                            for pg in opt.param_groups:
                                pg['lr'] = args.v2_emb_lr_post
                            embedding_schedulers.append(
                                torch.optim.lr_scheduler.ExponentialLR(
                                    opt, gamma=0.99))
                        print(f"[v2] Embedding lr -> {args.v2_emb_lr_post} "
                              f"(scheduler reset)")

                    agg.update_embeddings_ladder_triplet(
                        E_list, evidence, opt_list=embedding_opts, steps=steps)
                    for sch in embedding_schedulers:
                        sch.step()

                prev_ranked_neighbors = ranked_neighbors
                prev_ranked_dists     = ranked_dists

            # ================================================================
            # PAGANv3 — blended model+embedding affinity
            # ================================================================
            elif args.aggregation == 'p2p' and args.topo == 'paganv3':
                with torch.no_grad():
                    # 1. Sampling targets (identical 4-slot quota to v2)
                    neigh_lists = []
                    for i in range(args.num_spokes):
                        phys_list = (prev_ranked_neighbors[i].tolist()
                                     if prev_ranked_neighbors is not None else [])
                        targets_i = pagan_v3_protocol.get_sampling_targets(
                            rnd, i, args.k, phys_list, E_list[i])
                        neigh_lists.append(torch.tensor(
                            targets_i, device=aggregator_device,
                            dtype=torch.long))
 
                    # 2. Physical distance ranking
                    ranked_neighbors, ranked_dists = \
                        agg.rank_neighbors_by_model_distance(
                            node_states, neigh_lists)
 
                    # 3. Record EMA (identical to v2)
                    pagan_v3_protocol.record(rnd, ranked_neighbors, ranked_dists)
 
                    # 4. Blended aggregation
                    new_states = pagan_v3_protocol.run_topology_and_aggregate(
                        rnd, node_states, E_list, ranked_neighbors,
                        prev_ranked_neighbors)
                    node_states.copy_(new_states)
 
                # 5. Embedding update with rank-weighted triplet loss
                emb_frozen = (args.v2_freeze_embeddings
                              and rnd >= args.v2_warmup_rounds)
                if not emb_frozen and prev_ranked_neighbors is not None:
                    selection = [
                        torch.cat([ranked_neighbors[i],
                                   torch.tensor([i], device=aggregator_device,
                                                dtype=torch.long)])
                        for i in range(args.num_spokes)
                    ]
                    evidence = agg.gather_neighbor_ranklists_with_dists(
                        prev_ranked_neighbors, prev_ranked_dists,
                        chosen_neighbors=selection)
 
                    # Rank-weighted triplet evidence
                    evidence_weights = pagan_v3_protocol.get_evidence_weights(
                        evidence)
 
                    steps = (args.v2_emb_steps if rnd < args.v2_warmup_rounds
                             else max(1, args.v2_emb_steps // 2))
 
                    if rnd == args.v2_warmup_rounds and not args.v2_freeze_embeddings:
                        embedding_schedulers = []
                        for opt in embedding_opts:
                            for pg in opt.param_groups:
                                pg['lr'] = args.v2_emb_lr_post
                            embedding_schedulers.append(
                                torch.optim.lr_scheduler.ExponentialLR(
                                    opt, gamma=0.99))
                        print(f"[v3] Embedding lr -> {args.v2_emb_lr_post}")
 
                    agg.update_embeddings_ladder_triplet(
                        E_list, evidence,
                        opt_list=embedding_opts,
                        steps=steps,
                        trust_weights=evidence_weights)
 
                    for sch in embedding_schedulers:
                        sch.step()
 
                prev_ranked_neighbors = ranked_neighbors
                prev_ranked_dists     = ranked_dists


            elif args.aggregation == 'p2p' and args.topo == 'paganv4':
                with torch.no_grad():
                    # 1. Sampling targets (same 4-slot quota as v2)
                    neigh_lists = []
                    for i in range(args.num_spokes):
                        phys_list = (prev_ranked_neighbors[i].tolist()
                                     if prev_ranked_neighbors is not None else [])
                        targets_i = pagan_v4_protocol.get_sampling_targets(
                            rnd, i, args.k, phys_list, E_list[i])
                        neigh_lists.append(torch.tensor(
                            targets_i, device=aggregator_device, dtype=torch.long))
 
                    # 2. Physical distance ranking
                    ranked_neighbors, ranked_dists = \
                        agg.rank_neighbors_by_model_distance(node_states, neigh_lists)
 
                    # 3. Record: EMA update + trust recomputation for sampled pairs
                    pagan_v4_protocol.record(rnd, ranked_neighbors, ranked_dists)
 
                    # 4. Trust-gated aggregation
                    new_states = pagan_v4_protocol.run_topology_and_aggregate(
                        rnd, node_states, E_list, ranked_neighbors,
                        prev_ranked_neighbors)
                    node_states.copy_(new_states)
 
                # 5. Embedding update: trust × rank-weighted evidence
                emb_frozen = (args.v2_freeze_embeddings
                              and rnd >= args.v2_warmup_rounds)
                if not emb_frozen and prev_ranked_neighbors is not None:
                    selection = [
                        torch.cat([ranked_neighbors[i],
                                   torch.tensor([i], device=aggregator_device,
                                                dtype=torch.long)])
                        for i in range(args.num_spokes)
                    ]
                    evidence = agg.gather_neighbor_ranklists_with_dists(
                        prev_ranked_neighbors, prev_ranked_dists,
                        chosen_neighbors=selection)
 
                    # v4 combined weight: trust(i,j) / (rank + 1)
                    trust_weights = pagan_v4_protocol.get_trust_weights_for_evidence(
                        evidence)
 
                    steps = (args.v2_emb_steps if rnd < args.v2_warmup_rounds
                             else max(1, args.v2_emb_steps // 2))
 
                    if rnd == args.v2_warmup_rounds and not args.v2_freeze_embeddings:
                        embedding_schedulers = []
                        for opt in embedding_opts:
                            for pg in opt.param_groups:
                                pg['lr'] = args.v2_emb_lr_post
                            embedding_schedulers.append(
                                torch.optim.lr_scheduler.ExponentialLR(
                                    opt, gamma=0.99))
                        print(f"[v4] Embedding lr -> {args.v2_emb_lr_post} "
                              f"(scheduler reset)")
 
                    agg.update_embeddings_ladder_triplet(
                        E_list, evidence,
                        opt_list=embedding_opts,
                        steps=steps,
                        trust_weights=trust_weights)
                    for sch in embedding_schedulers:
                        sch.step()
 
                prev_ranked_neighbors = ranked_neighbors
                prev_ranked_dists     = ranked_dists


            # ── k-regular and DPFL paths (unchanged from main_0305) ───
            elif args.aggregation == 'p2p' and args.topo == 'dpfl':
                dpfl_topology = dpfl_selector.build_collaboration_graph(
                    node_states, dpfl_val_map,
                    net_name, inp_dim, out_dim,
                    budget_k=args.k, sample_ratio=0.2)
                W_dpfl = torch.zeros(
                    (args.num_spokes, args.num_spokes), device=aggregator_device)
                for i, neighbors in enumerate(dpfl_topology):
                    if len(neighbors) > 0:
                        W_dpfl[i, neighbors] = 1.0 / len(neighbors)
                    else:
                        W_dpfl[i, i] = 1.0
                node_states = agg.p2p_aggregation(node_states, W_dpfl).detach()
            
            elif args.aggregation == 'p2p' and args.topo == 'fedspd':
                pass  # training + aggregation already handled in run_round above
            
            elif args.aggregation == 'p2p' and args.topo == 'l2c':
                node_states = l2c_protocol.aggregate_and_update(
                    rnd,
                    node_states,
                    distributed_val_data,
                    distributed_val_label,
                    net_name, inp_dim, out_dim,
                )
            elif args.aggregation == 'p2p' and args.topo == 'federico':
                node_states = federico_protocol.aggregate_and_update(
                    rnd, node_states,
                    distributed_train_data, distributed_train_label,
                    net_name, inp_dim, out_dim, args.lr,
                )
    

            elif args.aggregation == 'p2p':
                if args.topo == 'k-regular-random':
                    _, W_local = agg.k_regular_random(
                        args.num_spokes, args.k, aggregator_device)
                elif args.topo == 'k-regular-clustered':
                    _, W_local = agg.k_regular_clustered(
                        args.num_spokes, node_to_cluster, args.k, aggregator_device)
                node_states = agg.p2p_aggregation(node_states, W_local).detach()

            # ================================================================
            # Evaluation
            # ================================================================
            if args.aggregation == 'p2p' and (rnd + 1) % args.eval_time == 0:

                # v2 embedding quality (all rounds including warmup)
                if args.topo == 'paganv2':
                    emb_stats = compute_emb_stats(
                        E_list, node_to_cluster, jsd_matrix)
                    emb_stats['round'] = rnd
                    metrics['v2_emb_stats'].append(emb_stats)
                    print(f"  [emb rnd {rnd:3d}] "
                          f"tp@5={emb_stats['tp_at_5']:.3f}  "
                          f"tp@10={emb_stats['tp_at_10']:.3f}  "
                          f"tp@15={emb_stats['tp_at_15']:.3f}  "
                          f"tp@20={emb_stats['tp_at_20']:.3f}  "
                          f"jsd_rho={emb_stats['mean_jsd_rho']:.3f}")

                if args.topo == 'paganv2' and rnd >= args.v2_warmup_rounds:
                    sm = pagan_v2_protocol.compute_soft_metrics(rnd)
                    metrics['v2_soft_metrics'].append(sm)
                    print(f"  [v2 soft] tp={sm.get('tp',0)}  "
                          f"fp={sm.get('fp',0)}  "
                          f"fn={sm.get('fn',0)}  "
                          f"self_w={sm.get('avg_self_weight',0):.3f}")

                # v3 embedding quality (all rounds including warmup)
                if args.topo == 'paganv3' and rnd >= args.v2_warmup_rounds:
                    sm = pagan_v3_protocol.compute_soft_metrics(rnd)
                    metrics['v3_soft_metrics'].append(sm)
 
                    b = sm.get('blended', {})
                    m = sm.get('model',   {})
                    e = sm.get('emb',     {})
                    print(f"  [v3 rnd {rnd:3d}] α={sm['alpha']:.3f}  "
                          f"blend: tp={b.get('tp',0)} fp={b.get('fp',0)}  "
                          f"model: tp={m.get('tp',0)} fp={m.get('fp',0)}  "
                          f"emb:   tp={e.get('tp',0)} fp={e.get('fp',0)}")
 
                if args.topo == 'paganv3':
                    emb_stats = compute_emb_stats(
                        E_list, node_to_cluster, jsd_matrix)
                    emb_stats['round'] = rnd
                    metrics['v3_emb_stats'].append(emb_stats)
                    print(f"  [v3 emb rnd {rnd:3d}] "
                          f"tp@5={emb_stats['tp_at_5']:.3f}  "
                          f"tp@10={emb_stats['tp_at_10']:.3f}  "
                          f"tp@15={emb_stats['tp_at_15']:.3f}")

                if args.topo == 'paganv4':
                    emb_stats = compute_emb_stats(
                        E_list, node_to_cluster, jsd_matrix)
                    emb_stats['round'] = rnd
                    metrics['v4_emb_stats'].append(emb_stats)
                    print(f"  [v4 emb rnd {rnd:3d}] "
                          f"tp@5={emb_stats['tp_at_5']:.3f}  "
                          f"tp@10={emb_stats['tp_at_10']:.3f}  "
                          f"tp@15={emb_stats['tp_at_15']:.3f}  "
                          f"tp@20={emb_stats['tp_at_20']:.3f}  "
                          f"jsd_rho={emb_stats['mean_jsd_rho']:.3f}")
 
                if args.topo == 'paganv4' and rnd >= args.v2_warmup_rounds:
                    sm = pagan_v4_protocol.compute_soft_metrics(rnd)
                    metrics['v4_soft_metrics'].append(sm)
                    print(f"  [v4 soft] tp={sm.get('tp',0)}  "
                          f"fp={sm.get('fp',0)}  "
                          f"fn={sm.get('fn',0)}  "
                          f"self_w={sm.get('avg_self_weight',0):.3f}  "
                          f"trust_mean={sm.get('trust_mean',0):.3f}")

                # Parallel eval
                if args.num_workers > 0 and worker_pool is not None:
                    weights_cpu = [w.detach().cpu() for w in node_states]
                    payload = [(weights_cpu[i],
                                local_test_pairs[i] if local_test_pairs else None)
                               for i in range(args.num_spokes)]
                    results  = worker_pool.map(eval_worker.evaluate_node, payload)
                    g_losses, g_accs, l_losses, l_accs = zip(*results)
                    g_accs   = list(g_accs);   g_losses = list(g_losses)
                    l_accs   = list(l_accs);   l_losses = list(l_losses)

                    print(f"[Round {rnd+1}] Local Acc range: "
                          f"[{min(g_accs):.3f}, {max(g_accs):.3f}]")
                    metrics['round'].append(rnd + 1)
                    metrics['global_acc'].append(g_accs)
                    if args.eval_local and local_test_pairs is not None:
                        metrics['local_acc'].append(l_accs)
                        utils.print_acc_values(
                            [a for a in l_accs if not math.isnan(a)])
                else:
                    utils.evaluate_and_log(
                        current_round=rnd + 1, metrics=metrics,
                        model_source=node_states, mode='p2p',
                        test_data=test_data, local_test_pairs=local_test_pairs,
                        device=aggregator_device, batch_size=batch_size,
                        dataset_name=args.dataset,
                        net_name=net_name, inp_dim=inp_dim, out_dim=out_dim,
                        eval_local=args.eval_local)

    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()

    # ── Save registry for v2 ─────────────────────────────────────────
    if args.topo == 'paganv2':
        pagan_v2_protocol.registry.save(filename + '_shadow')
        if pagan_v2_protocol.soft_metrics_log:
            metrics['v2_soft_metrics'] = pagan_v2_protocol.soft_metrics_log

    # ── Save registry for v3 ─────────────────────────────────────────
    if args.topo == 'paganv3':
        pagan_v3_protocol.registry.save(filename + '_shadow')
        if pagan_v3_protocol.soft_metrics_log:
            metrics['v3_soft_metrics'] = pagan_v3_protocol.soft_metrics_log
        print(f"[PAGANv3] Saved registry -> {filename}_shadow")


    if args.topo == 'paganv4':
        pagan_v4_protocol.registry.save(filename + '_shadow')
        if pagan_v4_protocol.soft_metrics_log:
            metrics['v4_soft_metrics'] = pagan_v4_protocol.soft_metrics_log
        metrics['v4_trust_snapshots'] = pagan_v4_protocol.trust_snapshots
        np.savez_compressed(
            filename + '_v4_trust',
            trust          = pagan_v4_protocol.trust,
            W_warmup       = pagan_v4_protocol.W_warmup,
            warmup_bucket  = pagan_v4_protocol.warmup_bucket,
            warmup_count   = pagan_v4_protocol.warmup_count,
            conflict_mask  = pagan_v4_protocol.conflict_mask)
        print(f"[PAGANv4] Saved trust state -> {filename}_v4_trust.npz")

    # ── Save metrics ──────────────────────────────────────────────────
    if metrics.get('local_split_stats'):
        for rec in metrics['local_split_stats']:
            rec.pop('train_counts', None)
            rec.pop('test_counts', None)

    with open(filename + '_metrics.json', 'w') as f:
        json.dump(metrics, f, separators=(',', ':'), indent=None)
    print("Training complete.")

def compute_emb_stats(E_list, ntc, jsd_matrix, ks=(5, 10, 15, 20), spearman_k=10):
    """
    Embedding quality metrics:
      - JSD Spearman@k (oracle: embedding order vs true distributional similarity)
      - Top-k precision at multiple k values (fraction of embedding top-k that are cluster-mates)
    """
    from scipy.stats import spearmanr
    N = len(E_list)

    per_rho_jsd = np.full(N, np.nan, dtype=np.float32)
    per_tp = {k_val: np.full(N, np.nan, dtype=np.float32) for k_val in ks}

    jsd_np = jsd_matrix.cpu().numpy() if hasattr(jsd_matrix, 'cpu') \
             else np.array(jsd_matrix)

    for i in range(N):
        # ── Embedding distances from i ────────────────────────────────
        Ei     = E_list[i].detach().cpu().float().numpy()
        my_emb = Ei[i]
        emb_d  = np.linalg.norm(Ei - my_emb, axis=1)
        emb_d[i] = np.inf
        emb_order = np.argsort(emb_d)

        # ── JSD ground-truth top-k (oracle) ──────────────────────────
        jsd_row  = jsd_np[i].copy()
        jsd_row[i] = np.inf
        jsd_order = np.argsort(jsd_row)
        jsd_topk  = jsd_order[:spearman_k].tolist()

        # ── JSD Spearman@k ────────────────────────────────────────────
        emb_ranks_of_jsd = [int(np.where(emb_order == j)[0][0])
                             for j in jsd_topk]
        if len(emb_ranks_of_jsd) >= 2:
            rho, _ = spearmanr(np.arange(len(emb_ranks_of_jsd)),
                                emb_ranks_of_jsd)
            if not np.isnan(rho):
                per_rho_jsd[i] = float(rho)

        # ── Top-k precision at multiple k ─────────────────────────────
        my_c = ntc[i]
        for k_val in ks:
            top_k_nodes = emb_order[:k_val].tolist()
            tp = sum(1 for j in top_k_nodes if ntc[j] == my_c)
            per_tp[k_val][i] = tp / k_val

    result = {'mean_jsd_rho': float(np.nanmean(per_rho_jsd))}
    for k_val in ks:
        result[f'tp_at_{k_val}'] = float(np.nanmean(per_tp[k_val]))
    return result


if __name__ == "__main__":
    args = parse_args()
    main(args)