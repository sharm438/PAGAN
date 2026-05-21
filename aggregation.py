import torch
import numpy as np
import pdb
from typing import Optional
import scipy.stats

import random
import torch
import torch.nn.functional as F

import bisect 

import utils as utils 
import copy

from sklearn.cluster import KMeans

from collections import defaultdict
import math

class PAGANProtocol:
    """
    PAGAN: Peer-to-peer Aggregation via Graph Attention Networks.
    Architecture:
        - Batch Hyperfocus (Queue Management)
        - Ordinal Topology: Uses Moving Average Ranks instead of Float Distances.
        - Set-to-Set Steady State: Averages {Candidate, Alibi 1, Alibi 2} to prevent Runaway Blobs.
        - Organic Soft Merge: Weight = 1 / MovingAverageRank.
    """
    def __init__(self, num_nodes, device, node_to_cluster=None):
        self.num_nodes = num_nodes
        self.device = device
        self.node_to_cluster = node_to_cluster
        self.DEBUG_NODE = 0 
        
        self.agg_weights = [{i: 1.0} for i in range(num_nodes)]
        
        # Tribe Rank History: {friend_id: [list of W ranks]}
        self.RANK_WINDOW = 3
        self.tribe_rank_history = [defaultdict(lambda: [999]*self.RANK_WINDOW) for _ in range(num_nodes)]
        
        # Probation Dict: Holds Candidates AND their Alibis' rank histories
        self.probation_dict = [{} for _ in range(num_nodes)]
        self.cached_post_local = None 
        
        # --- The Knobs ---
        self.HYPERFOCUS_CAP = 2          
        self.PROBATION_DURATION = 4      
        self.ALIBI_COUNT = 2             
        self.DEFAULT_BAD_RANK = 20  

        self.WARMUP_ROUNDS = 50
        self.bootstrapped = False     
    
    def get_sampling_targets(self, rnd, node_i, k_total, physical_rank_list, E_list_i):
        targets = set()
        
        # 1. The Tribe (Empty during warmup)
        tribe = list(self.tribe_rank_history[node_i].keys())
        for f in tribe: targets.add(f)
            
        # 2. Active Probationers & Their Alibis (Empty during warmup)
        for p, stats in self.probation_dict[node_i].items():
            targets.add(p)
            for alibi in stats['alibis']:
                if len(targets) < k_total: targets.add(alibi)
                    
        # --- THE SIMPLE QUOTA LOGIC ---
        phys_limit = 3
        emb_limit = 3 if rnd < self.WARMUP_ROUNDS else 10
        
        # 3. Forced Physical Watch (Top 3 outside tribe)
        added_phys = 0
        for cand in physical_rank_list:
            if cand != node_i and cand not in tribe and cand not in targets:
                targets.add(cand)
                added_phys += 1
                if added_phys >= phys_limit or len(targets) >= k_total:
                    break 
                
        # 4. Exploit Embeddings (The Scout)
        added_emb = 0
        if len(targets) < k_total:
            my_emb = E_list_i[node_i].unsqueeze(0)
            emb_sorted = torch.argsort(torch.norm(E_list_i - my_emb, dim=1)).tolist()
            for cand in emb_sorted:
                if cand != node_i and cand not in targets:
                    targets.add(cand)
                    added_emb += 1
                if added_emb >= emb_limit or len(targets) >= k_total: 
                    break

        # 5. Random Explore (Fills the rest: 14 during warmup, remainder post-warmup)
        all_nodes = list(range(self.num_nodes))
        random.shuffle(all_nodes)
        for cand in all_nodes:
            if len(targets) >= k_total: break
            if cand != node_i and cand not in targets:
                targets.add(cand)

        return list(targets)

    def run_topology_and_aggregate(self, rnd, node_states, E_list, ranked_neighbors, prev_ranked_neighbors):
        if self.cached_post_local is None:
            self.cached_post_local = node_states.detach().clone()
            return node_states.clone()

        # --- WARMUP PHASE ---
        if rnd < self.WARMUP_ROUNDS:
            # Nodes do not mix models or accept friends yet. Just let embeddings learn.
            if self.node_to_cluster is not None: self.compute_diagnostics(rnd, E_list)
            return node_states.clone() 

        # --- THE BOOTSTRAP (Round 5) ---
        if rnd == self.WARMUP_ROUNDS and not self.bootstrapped:
            print("\n[BOOTSTRAP] Warmup complete. Initializing Rank-0 friends for all nodes.")
            for i in range(self.num_nodes):
                if len(ranked_neighbors[i]) > 0:
                    best_friend = ranked_neighbors[i][0].item()
                    # Initialize them with a perfect rank history
                    self.tribe_rank_history[i][best_friend] = [0] * self.RANK_WINDOW
            self.bootstrapped = True

        # --- NORMAL PROTOCOL PHASE ---
        dn = self.DEBUG_NODE
        dn_tribe = list(self.tribe_rank_history[dn].keys())
        dn_weights = {f: round(w, 3) for f, w in self.agg_weights[dn].items() if w > 0.01 and f != dn}
        
        print(f"\n================ ROUND {rnd} ================")
        print(f"[DEBUG Node {dn}] Tribe (Size {len(dn_tribe)}): {dn_tribe}")
        print(f"[DEBUG Node {dn}] Edge Weights: {dn_weights}")

        self._manage_queue_and_nominate(E_list, ranked_neighbors)
        self._evaluate_probationers(rnd, ranked_neighbors, prev_ranked_neighbors)
        self._update_tribe_weights_and_contract(rnd, ranked_neighbors)
        
        new_node_states = self._execute_aggregation(node_states)
        self.cached_post_local = node_states.detach().clone()
        
        if self.node_to_cluster is not None:
            self.compute_diagnostics(rnd, E_list)
            
        return new_node_states

    def _manage_queue_and_nominate(self, E_list, ranked_neighbors):
        for i in range(self.num_nodes):
            if len(self.probation_dict[i]) > 0: continue # Wait for desk to clear
            
            cands_added = 0
            tribe = list(self.tribe_rank_history[i].keys())
            my_ranks = ranked_neighbors[i].tolist()
            
            # Target 1: #1 Physical Stranger
            for cand in my_ranks:
                if cand != i and cand not in tribe:
                    self._open_case(i, cand, "Physical")
                    cands_added += 1
                    break
            
            # Target 2: Scout Recommendation
            my_emb = E_list[i][i].unsqueeze(0)
            emb_sorted = torch.argsort(torch.norm(E_list[i] - my_emb, dim=1)).tolist()
            for idx, cand in enumerate(emb_sorted):
                if cands_added >= self.HYPERFOCUS_CAP: break
                if cand != i and cand not in tribe and cand not in self.probation_dict[i]:
                    self._open_case(i, cand, "Scout")
                    cands_added += 1
                    #print ("Scout rankings: ", emb_sorted[:idx+1])

    def _open_case(self, node_i, cand, reason):
        self.probation_dict[node_i][cand] = {
            'age': 0, 
            'alibis': [], 
            'cand_history': [], 
            'alibi_histories': defaultdict(list)
        }
        if node_i == self.DEBUG_NODE: 
            print(f"[DEBUG Rnd] Node {node_i} NOMINATION: Node {cand} ({reason}).")

    def _evaluate_probationers(self, rnd, ranked_neighbors, prev_ranked_neighbors):
        for i in range(self.num_nodes):
            cands_to_remove = []
            my_ranks = ranked_neighbors[i].tolist()
            tribe_size = len(self.tribe_rank_history[i])
            
            # The Dynamic Anchor: Tribe Size + 1 (Min 3)
            acceptance_threshold = max(3.0, float(tribe_size))
            
            for cand, stats in self.probation_dict[i].items():
                stats['age'] += 1
                
                # Fetch Alibis asynchronously on Round 1 of Probation
                if len(stats['alibis']) == 0 and prev_ranked_neighbors is not None:
                    cand_list = prev_ranked_neighbors[cand].tolist()
                    stats['alibis'] = [x for x in cand_list if x != cand and x != i][:self.ALIBI_COUNT]
                
                # Track Physical Ranks
                try: cand_rank = my_ranks.index(cand)
                except ValueError: cand_rank = self.DEFAULT_BAD_RANK
                stats['cand_history'].append(cand_rank)
                
                for alibi in stats['alibis']:
                    try: alibi_rank = my_ranks.index(alibi)
                    except ValueError: alibi_rank = self.DEFAULT_BAD_RANK
                    stats['alibi_histories'][alibi].append(alibi_rank)

                if i == self.DEBUG_NODE:
                    print(f"[DEBUG Rnd {rnd}] Node {i} Bouncer: Cand {cand} (Age {stats['age']}) | Ranks -> Cand: {cand_rank}, Alibis: {[my_ranks.index(a) if a in my_ranks else 999 for a in stats['alibis']]}")

                # The Verdict
                # if stats['age'] >= self.PROBATION_DURATION:
                #     # Calculate Set Average
                #     avg_cand = sum(stats['cand_history']) / max(1, len(stats['cand_history']))
                #     alibi_avgs = [sum(h)/max(1, len(h)) for h in stats['alibi_histories'].values()]
                    
                #     set_elements = [avg_cand] + alibi_avgs
                #     set_avg_rank = sum(set_elements) / max(1, len(set_elements))
                    
                #     if i == self.DEBUG_NODE:
                #         print(f"   -> Set Avg Rank: {set_avg_rank:.2f} | Threshold: {acceptance_threshold:.2f}")

                #     if set_avg_rank <= acceptance_threshold:
                #         # ACCEPT
                #         self.tribe_rank_history[i][cand] = [int(avg_cand)] * self.RANK_WINDOW
                #         if i == self.DEBUG_NODE: print(f"   >>> [EXPANSION] Node {i} ACCEPTED Node {cand}! <<<")
                #     else:
                #         if i == self.DEBUG_NODE: print(f"   --- [REJECTION] Node {i} REJECTED Node {cand} (Failed Set-Average). ---")
                    
                #     cands_to_remove.append(cand)
                # --- THE VERDICT (Age 4) ---
                if stats['age'] >= self.PROBATION_DURATION:
                    
                    # Calculate Averages
                    avg_cand = sum(stats['cand_history']) / max(1, len(stats['cand_history']))
                    alibi_avgs = [sum(h)/max(1, len(h)) for h in stats['alibi_histories'].values()]
                    
                    set_elements = [avg_cand] + alibi_avgs
                    set_avg_rank = sum(set_elements) / max(1, len(set_elements))
                    
                    # --- OPTION 2: THE ANNEALING THRESHOLD ---
                    if tribe_size < 2:
                        buffer = 3.0
                    elif tribe_size < 4:
                        buffer = 2.0
                    elif tribe_size < 6:
                        buffer = 1.0
                    else:
                        buffer = 0.0  # The Brick Wall (Steady State)
                        
                    acceptance_threshold = float(tribe_size) + buffer
                    
                    if i == self.DEBUG_NODE:
                        print(f"   -> Set Avg: {set_avg_rank:.2f} | Tribe Size: {tribe_size} | Dynamic Threshold: {acceptance_threshold:.2f}")

                    if set_avg_rank <= acceptance_threshold:
                        # ACCEPT
                        self.tribe_rank_history[i][cand] = [int(avg_cand)] * self.RANK_WINDOW
                        if i == self.DEBUG_NODE: print(f"   >>> [EXPANSION] Node {i} ACCEPTED Node {cand}! <<<")
                    else:
                        if i == self.DEBUG_NODE: print(f"   --- [REJECTION] Node {i} REJECTED Node {cand}. ---")
                    
                    cands_to_remove.append(cand)
            
            for c in cands_to_remove: del self.probation_dict[i][c]

    def _update_tribe_weights_and_contract(self, rnd, ranked_neighbors):
        for i in range(self.num_nodes):
            tribe = list(self.tribe_rank_history[i].keys())
            my_ranks = ranked_neighbors[i].tolist()
            tribe_size = len(tribe)
            
            # The Demotion Threshold: Tribe Size + 2 (Min 5)
            demotion_threshold = max(5.0, float(tribe_size + 1))
            
            raw_weights = {i: 1.0} 
            friends_to_demote = []
            
            for f in tribe:
                try: current_rank = my_ranks.index(f)
                except ValueError: current_rank = self.DEFAULT_BAD_RANK
                
                # Update Rolling History
                self.tribe_rank_history[i][f].pop(0)
                self.tribe_rank_history[i][f].append(current_rank)
                
                # Check for Demotion
                avg_rank = sum(self.tribe_rank_history[i][f]) / self.RANK_WINDOW
                
                if avg_rank > demotion_threshold:
                    if i == self.DEBUG_NODE: print(f"   !!! [DEMOTION] Friend {f} drifted to Avg Rank {avg_rank:.2f} (Threshold {demotion_threshold:.2f}). Stripping Tribe status! !!!")
                    friends_to_demote.append(f)
                    continue 
                
                # Organic Weight
                raw_weights[f] = 1.0 / math.sqrt(max(1.0, avg_rank)) 
                
            # Execute Demotions (Ex-friends back to probation)
            for f in friends_to_demote:
                del self.tribe_rank_history[i][f]
                self._open_case(i, f, "Ex-Friend Demotion")

            # Normalize weights
            total_mass = sum(raw_weights.values())
            self.agg_weights[i] = {node: (w / total_mass) for node, w in raw_weights.items()}

    def _execute_aggregation(self, node_states):
        new_states = torch.zeros_like(node_states)
        for i in range(self.num_nodes):
            mixed_w = torch.zeros_like(node_states[i])
            for f, pi_f in self.agg_weights[i].items():
                mixed_w += pi_f * node_states[f]
            new_states[i] = mixed_w
        return new_states

    def compute_diagnostics(self, rnd, E_list=None):
        tp, fp, fn, tn = 0, 0, 0, 0
        cluster_counts = defaultdict(int)
        for c in self.node_to_cluster: cluster_counts[c] += 1
            
        for i in range(self.num_nodes):
            my_c = self.node_to_cluster[i]
            ideal_friends = cluster_counts[my_c] - 1
            
            actual_friends = list(self.tribe_rank_history[i].keys())
            matches = sum([1 for f in actual_friends if self.node_to_cluster[f] == my_c])
            mismatches = len(actual_friends) - matches
            
            tp += matches
            fp += mismatches
            fn += max(0, ideal_friends - matches)
            tn += (self.num_nodes - 1 - ideal_friends) - mismatches
            
        print(f"--- Round {rnd} Diagnostics ---")
        print(f"Graph Metrics -> TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
        avg_self = sum([self.agg_weights[i].get(i, 0.0) for i in range(self.num_nodes)]) / self.num_nodes
        print(f"Avg Self-Weight: {avg_self:.3f}")
        
        # --- SCOUT LATENT SPACE DIAGNOSTIC ---
        if E_list is not None:
            dn = self.DEBUG_NODE
            my_emb = E_list[dn][dn].unsqueeze(0)
            my_c = self.node_to_cluster[dn]
            
            # Compute L2 Distances to all other nodes
            dists = []
            for j in range(self.num_nodes):
                if j == dn: continue
                their_emb = E_list[dn][j].unsqueeze(0)
                dist = torch.norm(their_emb - my_emb, p=2).item()
                dists.append((dist, j))
                
            # Sort by lowest distance
            dists.sort(key=lambda x: x[0])
            top_15 = dists[:15]
            
            print(f"[DEBUG Node {dn} (Cluster {my_c})] Scout's Top-15 Latent Predictions:")
            
            prediction_str = []
            for d, j in top_15:
                j_c = self.node_to_cluster[j]
                # Mark true friends with a '*' for easy reading
                match = "*" if j_c == my_c else ""
                prediction_str.append(f"{j}{match}({d:.2f})")
                
            print("   -> " + ", ".join(prediction_str))
            
            # Recreate Precision@10 on the fly just for the printout
            hits = sum([1 for d, j in top_15[:10] if self.node_to_cluster[j] == my_c])
            print(f"   -> Scout Precision@10: {hits}/10")
            
        print("-" * 30)

class PAGANSelector:
    """
    PAGAN V5: Centroid Verification Protocol.
    Replaces Traffic Light/Knee logic with Relative Trust Anchors.
    """
    def __init__(self, num_nodes, device, node_to_cluster=None):
        self.num_nodes = num_nodes
        self.device = device
        self.node_to_cluster = node_to_cluster
        
        # --- State ---
        self.verified_friends = [set() for _ in range(num_nodes)]
        
        # Probation: {node_id: rounds_remaining}
        self.probation_queue = [{} for _ in range(num_nodes)]
        
        # Stats: {node_id: {'pass': 0, 'fail': 0}}
        self.trust_stats = [{} for _ in range(num_nodes)]
        
        # Constants
        self.PHASE_0_END = 5   # Ghost Training
        self.PHASE_1_END = 10  # Seeding
        
        self.TRUST_THRESHOLD = 3#4 # pass - fail > 4
        self.FAIL_THRESHOLD = 4  # fail - pass > 4
        self.PROBATION_DURATION = 5 #8 
        self. ALPHA= 1.2 
        
        # Diagnostics
        self.metrics = {'tp': 0, 'fp': 0}

        self.tuning_stats = {
            'ratio_tp': [], # Friends who Passed (Value < 1.0)
            'ratio_fn': [], # Friends who Failed (Value >= 1.0) -> TARGET TO REDUCE
            'ratio_fp': [], # Strangers who Passed (Value < 1.0) -> TARGET TO AVOID
            'ratio_tn': [], # Strangers who Failed (Value >= 1.0) -> THE SAFETY MARGIN
        }

    def get_forcing_list(self):
        """
        Returns list of nodes to FORCE sample:
        1. Verified Friends (Maintenance - needed to update centroids)
        2. Probation Nodes (Vetting - needed to run distance tests)
        """
        res = []
        for i in range(self.num_nodes):
            forced = set()
            forced.update(self.verified_friends[i])
            forced.update(self.probation_queue[i].keys())
            
            if len(forced) > 0:
                t = torch.tensor(list(forced), device=self.device, dtype=torch.long)
            else:
                t = torch.tensor([], device=self.device, dtype=torch.long)
            res.append(t)
        return res

    def run_selection(self, rnd, ranked_neighbors, node_states, E_list, velocities=None):
        """
        Executes PAGAN V5 Logic.
        Args:
            node_states: Tensor [N, ModelDim] (Weights)
            E_list: List of Tensors or Tensor [N, N, D] (Embeddings)
        """
        final_topology = []
        self.metrics = {'tp': 0, 'fp': 0}
        self.tuning_stats = {'ratio_tp': [], 'ratio_fn': [], 'ratio_fp': [], 'ratio_tn': []}
        # Ensure E is accessible as tensor
        E_tensor = torch.stack(E_list).detach()

        for i in range(self.num_nodes):
            neigh = ranked_neighbors[i] # Tensor [K], sorted by model distance
            r_floor = velocities[i] if (velocities is not None and len(velocities) > i) else 1e-4
            # --- Phases 0 & 1 (Warmup / Seeding) ---
            if rnd < self.PHASE_1_END:
                # 1. Force Top-3 for 1 Round (Quick Churn)
                # This helps find the true Rank 0 quickly.
                if len(neigh) > 0:
                    top_k_warmup = neigh[:3].tolist()
                    for cand in top_k_warmup:
                        if cand == i: continue
                        #if cand not in self.probation_queue[i]:
                        self.probation_queue[i][cand] = 2 # 2 Round duration
                            
                        if cand not in self.trust_stats[i]:
                            self.trust_stats[i][cand] = {'pass': 0, 'fail': 0}
                    #if i == 0:
                        
                # 2. Topology Selection
                if rnd < self.PHASE_0_END:
                    # Phase 0: Ghost (No Aggregation)
                    final_topology.append(torch.tensor([], device=self.device, dtype=torch.long))
                else:
                    # Phase 1: Rank 0 (Seed)
                    if len(neigh) > 0:
                        top_node = neigh[0].item()
                        # Auto-seed Rank 0
                        self.verified_friends[i].add(top_node)
                        if top_node not in self.trust_stats[i]:
                             self.trust_stats[i][top_node] = {'pass': 0, 'fail': 0}
                        final_topology.append(neigh[:1])
                    else:
                        final_topology.append(torch.tensor([], device=self.device, dtype=torch.long))
                
                # Cleanup expired probation
                self._cleanup_probation(i)
                if i == 0:
                    print(rnd, "ranked neighbors", ranked_neighbors[i])
                    print(rnd, "probation queue", self.probation_queue[i])
                    print(rnd, "verified friends", self.verified_friends[i])
                continue

            # --- Phase 2: Centroid Verification ---
            
            # 1. Add New Candidates to Probation
            # Scan ranked list for top 2 candidates we don't know yet
            added = 0
            for cand in neigh.tolist():
                if added >= 3: break
                if cand == i: continue
                if cand in self.verified_friends[i]: continue
                #if cand in self.probation_queue[i] and self.probation_queue[i][cand] > 1: continue
                
                # Check blacklist history
                # stats = self.trust_stats[i].get(cand, {'pass':0, 'fail':0})
                # if stats['fail'] - stats['pass'] > self.FAIL_THRESHOLD:
                #     continue 
                
                # Add to probation
                self.probation_queue[i][cand] = self.PROBATION_DURATION
                if cand not in self.trust_stats[i]:
                    self.trust_stats[i][cand] = {'pass': 0, 'fail': 0}
                added += 1

            # 2. Form Clusters from Verified Friends
            friends = list(self.verified_friends[i])
            clusters = [] 
            
            if len(friends) > 0:
                friend_indices = torch.tensor(friends, device=self.device, dtype=torch.long)
                friend_wts = node_states[friend_indices] 
                friend_embs = E_tensor[i, friend_indices] 
                
                my_w = node_states[i]
                
                # Clustering Logic
                N = len(friends)
                if N < 4:
                    # Singletons: Each friend is a cluster (High fidelity)
                    for f_idx in range(N):
                        c_w = friend_wts[f_idx]
                        r = torch.norm(my_w - c_w).item()
                        clusters.append({'w': c_w, 'r': r})
                else:
                    # K=3 Grouping
                    k_c = 3
                    embs_np = friend_embs.cpu().numpy()
                    kmeans = KMeans(n_clusters=k_c, n_init='auto').fit(embs_np)
                    labels = kmeans.labels_
                    
                    for c_id in range(k_c):
                        mask = (labels == c_id)
                        if not np.any(mask): continue
                        
                        cluster_wts = friend_wts[torch.tensor(mask, device=self.device)]
                        
                        # Centroid = Mean (Weight Integrity)
                        cent_w = torch.mean(cluster_wts, dim=0)
                        
                        # Radius = Median Distance from Me (Robust Scale)
                        dists_to_me = torch.norm(cluster_wts - my_w, dim=1)
                        radius = torch.max(dists_to_me).item()
                        
                        clusters.append({'w': cent_w, 'r': radius})

            # 3. Test Candidates
            # Test everyone in the current sampled list that we care about (Friends + Probation)
            #relevant = self.verified_friends[i].union(set(self.probation_queue[i].keys()))
            
            # Debug helpers
            my_c = self.node_to_cluster[i] if self.node_to_cluster is not None else -1
            #if rnd > 10: pdb.set_trace()
            for cand in neigh.tolist():
                
                cand_w = node_states[cand]
                my_w = node_states[i]
                d_me = torch.norm(my_w - cand_w).item()
                
                passed = False
                best_ratio_for_cand = float('inf')
                if len(clusters) == 0:
                    # No friends yet (Phase transition edge case).
                    pass 
                else:
                    # Test against ALL clusters. Pass if accepted by ANY.
                    for c in clusters:
                        d_cent = torch.norm(c['w'] - cand_w).item()
                        # R is the raw median radius. Limit is Alpha * R.
                        # We want the ratio against R directly to see what Alpha should be.
                        # Ratio = Distance / Radius
                        # If Ratio < Alpha, we pass.
                        r_trust = c['r']
                        r = max(r_trust, r_floor, 1e-6)

                        if i == 0 and r_floor > r_trust and r_trust < 1e-3:
                            print(f"[DEBUG Node 0] IID Protection Active! r_trust={r_trust:.5f} < r_floor={r_floor:.5f}. "
                              f"New Radius={r_effective:.5f}")
                        
                        # Calculate ratios relative to the RAW radius (before Alpha)
                        # This tells us "How many Radii away is this node?"
                        r_me   = d_me / r
                        r_cent = d_cent / r
                        
                        # The test allows passing if EITHER is good
                        current_ratio = min(r_me, r_cent)
                        
                        if current_ratio < best_ratio_for_cand:
                            best_ratio_for_cand = current_ratio
                            
                        # Check against actual limit (Alpha * r)
                        limit = self.ALPHA * r
                        if d_me < self.ALPHA * r or d_cent < r:
                            passed = True
                            # We don't break yet if we want to find absolute best ratio? 
                            # Optimization: Break if passed is True, but we already calculated best_ratio
                            break
                # --- Tuning Data Logging ---
                cand_c = self.node_to_cluster[cand] if self.node_to_cluster is not None else -2
                is_same_cluster = (cand_c == my_c)
                
                # We only log if we actually computed a ratio (clusters existed)
                if best_ratio_for_cand != float('inf'):
                    if is_same_cluster:
                        if passed: self.tuning_stats['ratio_tp'].append(best_ratio_for_cand)
                        else:      self.tuning_stats['ratio_fn'].append(best_ratio_for_cand)
                    else:
                        if passed: self.tuning_stats['ratio_fp'].append(best_ratio_for_cand)
                        else:      self.tuning_stats['ratio_tn'].append(best_ratio_for_cand)    
                               
                if cand not in self.trust_stats[i]:
                    self.trust_stats[i][cand] = {'pass': 0, 'fail': 0}
                # Update Stats
                stats = self.trust_stats[i][cand]
                
                # --- PDB TRAPS ---
                cand_c = self.node_to_cluster[cand] if self.node_to_cluster is not None else -2
                is_fp = (cand_c != my_c)
                
                if passed:
                    stats['pass'] += 1
                    if is_fp:
                        # FP Passed! 
                        # print(f"[DEBUG] FP Passed Node {i}->{cand}. D_me={d_me:.3f}")
                        # pdb.set_trace()
                        pass
                else:
                    stats['fail'] += (r_me - self.ALPHA)
                    if not is_fp:
                        # TP Failed!
                        # print(f"[DEBUG] TP Failed Node {i}->{cand}. D_me={d_me:.3f}")
                        # pdb.set_trace()
                        pass

                # Decision Logic
                score = stats['pass'] - stats['fail']
                
                if score > self.TRUST_THRESHOLD:
                    self.verified_friends[i].add(cand)
                    if cand in self.probation_queue[i]: del self.probation_queue[i][cand]
                    
                # elif score < -self.FAIL_THRESHOLD: # Stop Forcing
                #     if cand in self.verified_friends[i]: self.verified_friends[i].remove(cand)
                #     if cand in self.probation_queue[i]: del self.probation_queue[i][cand]
                    
                # else:
                #     # Continue Probation
                #     # if i == 0:
                #     #     print(rnd, self.probation_queue[i])
                #     #     pdb.set_trace()
                #     if cand in self.probation_queue[i]:
                #         self.probation_queue[i][cand] -= 1
                #         if self.probation_queue[i][cand] <= 0:
                #             del self.probation_queue[i][cand] # Timeout
                    
            # 4. Final Topology: Verified Friends Only
            # (We only aggregate fully trusted nodes)
            final = list(self.verified_friends[i])
            final_tensor = torch.tensor(final, device=self.device, dtype=torch.long)
            final_topology.append(final_tensor)
            # Metrics
            if self.node_to_cluster is not None and len(final) > 0:
                matches = [1 if self.node_to_cluster[n] == my_c else 0 for n in final]
                self.metrics['tp'] += sum(matches)
                self.metrics['fp'] += (len(matches) - sum(matches))
            
            self._cleanup_probation(i)

            if i == 0:
                print(f"\n--- Round {rnd} Debug [Node 0] ---")
                print(f"Ranked (Top 10): {neigh[:10].tolist()}")
                print(f"Verified: {self.verified_friends[i]}")
                
                print("Probation & Exploration Status:")
                # Check everyone in probation + top 3 ranked (to see why they aren't entering)
                targets = list(self.probation_queue[i].keys())
                
                for cand in targets:
                    cand_w = node_states[cand]
                    d_me = torch.norm(node_states[i] - cand_w).item()
                    
                    # Calculate Best Ratio (The "Why")
                    best_ratio = float('inf')
                    radius_used = 0.0
                    
                    if len(clusters) > 0:
                        for c in clusters:
                            r = max(c['r'], 1e-6)
                            d_cent = torch.norm(c['w'] - cand_w).item()
                            
                            # Check both approaches (Me vs Centroid)
                            # ratio = Distance / Radius
                            r_me = d_me / r
                            r_cent = d_cent / r
                            
                            # We take the best chance they had
                            current_best = min(r_me, r_cent)
                            if current_best < best_ratio:
                                best_ratio = current_best
                                radius_used = r
                    
                    stats = self.trust_stats[i].get(cand, {'pass':0, 'fail':0})
                    score = stats['pass'] - stats['fail']
                    
                    # Visual Indicator
                    status = "✅ PASS" if best_ratio < self.ALPHA else f"❌ FAIL (Need < {self.ALPHA})"
                    
                    print(f"   Node {cand}: Score={score} ({stats['pass']}|{stats['fail']}) "
                          f"| Dist={d_me:.3f} | Radius={radius_used:.3f} "
                          f"| Ratio={best_ratio:.2f}x -> {status}")
                print("----------------------------------\n")
        return final_topology

    def _cleanup_probation(self, i):
        to_remove = []
        for cand in self.probation_queue[i]:
            self.probation_queue[i][cand] -= 1
            if self.probation_queue[i][cand] <= 0:
                to_remove.append(cand)
        for cand in to_remove:
            del self.probation_queue[i][cand]

def compute_graph_stats(agg_neighbors, node_to_cluster):
    """
    Computes the TP/FP/FN/TN based on the actual nodes selected for aggregation.
    """
    tp = 0
    fp = 0
    fn = 0
    
    # Calculate global cluster sizes to know ideal counts
    # (Assuming all nodes in a cluster should be connected to all others)
    cluster_counts = {}
    for c in node_to_cluster:
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
        
    for i, neighbors in enumerate(agg_neighbors):
        my_c = node_to_cluster[i]
        ideal_neighbors = cluster_counts[my_c] - 1 # Exclude self
        
        # 1. Analyze Neighbors held
        if len(neighbors) == 0:
            current_tp = 0
            current_fp = 0
        else:
            # Check matches
            matches = [1 if node_to_cluster[n.item()] == my_c else 0 for n in neighbors]
            current_tp = sum(matches)
            current_fp = len(matches) - current_tp
            
        # 2. Derive Misses
        # FN = (Everyone I SHOULD have) - (The friends I actually have)
        current_fn = max(0, ideal_neighbors - current_tp)
        
        # 3. Derive TN (Rest of the world)
        # Total nodes - 1 (self) - Ideal_Friends - False_Positives
        # (This is less critical but completes the matrix)
        
        tp += current_tp
        fp += current_fp
        fn += current_fn
        
    return tp, fp, fn
    
class DPFLSelector:
    """
    Implements the Greedy Graph Construction (GGC) from DPFL (Kharrat et al., 2025).
    Each client selects neighbors greedily by maximizing the reduction in local validation loss.
    """
    def __init__(self, num_nodes, device):
        self.num_nodes = num_nodes
        self.device = device
        self.selected_neighbors = [[] for _ in range(num_nodes)]

    def build_collaboration_graph(self, 
                                  node_states,       # List[Tensor] or stacked Tensor of model weights
                                  val_data_map,      # Dict {node_id: (X_val, y_val)}
                                  net_name, inp_dim, out_dim, 
                                  budget_k=10,       # Max neighbors (budget)
                                  sample_ratio=0.5): # Optimization: Sample % of peers to evaluate if N is large
        """
        Runs the Constrained Greedy Algorithm for each node.
        
        Note: This involves N * K * (Candidate_Pool) inferences. 
        To keep it feasible, we use a small val_batch and can subsample candidates.
        """

        new_topology = []
        
        # Ensure node_states is accessible
        if isinstance(node_states, list):
            all_weights = torch.stack(node_states)
        else:
            all_weights = node_states

        criterion = torch.nn.CrossEntropyLoss()

        for i in range(self.num_nodes):
            # 1. Get Val Data for Node i
            if i not in val_data_map or val_data_map[i] is None:
                # Fallback: maintain self-loop only if no val data
                new_topology.append(torch.tensor([i], device=self.device))
                continue
                
            X_val, y_val = val_data_map[i]
            # Optimization: Use a small fixed batch for the greedy check to save time
            val_batch_size = min(len(X_val), 64) 
            X_v, y_v = X_val[:val_batch_size].to(self.device), y_val[:val_batch_size].to(self.device)

            # 2. Candidate Set (All other nodes)
            # In massive networks, DPFL might sample candidates. 
            candidates = [idx for idx in range(self.num_nodes) if idx != i]
            if sample_ratio < 1.0:
                import random
                n_sample = max(budget_k, int(len(candidates) * sample_ratio))
                candidates = random.sample(candidates, n_sample)

            # 3. Greedy Selection
            # Start with Self
            current_selection = [i]
            
            # Helper: Evaluate Loss of Averaged Model
            def evaluate_set(indices):
                # Mean of weights
                selected_wts = all_weights[indices]
                avg_wt = torch.mean(selected_wts, dim=0)
                
                # Reconstruct
                temp_model = utils.vec_to_model(avg_wt, net_name, inp_dim, out_dim, self.device)
                temp_model.eval()
                
                with torch.no_grad():
                    out = temp_model(X_v)
                    loss = criterion(out, y_v).item()
                return loss

            current_loss = evaluate_set(current_selection)
            
            # Iteratively add best candidate
            for _ in range(budget_k):
                best_cand = -1
                best_new_loss = current_loss
                
                for cand in candidates:
                    if cand in current_selection: continue
                    
                    # Test adding cand
                    trial_loss = evaluate_set(current_selection + [cand])
                    
                    if trial_loss < best_new_loss:
                        best_new_loss = trial_loss
                        best_cand = cand
                
                # If we found a beneficial neighbor, add them
                if best_cand != -1:
                    current_selection.append(best_cand)
                    current_loss = best_new_loss
                    # Remove from candidates so we don't check again
                    # (Strictly not needed if we check 'in current_selection', but efficient)
                else:
                    # Submodularity property: If no single addition helps, stop early
                    break
            
            new_topology.append(torch.tensor(current_selection, device=self.device))

        return new_topology

def sample_candidates_disjoint(
    rnd,
    E_list, 
    k_exploit,      # args.k_soft
    n_random,       # args.k_rand
    node_to_cluster=None,
    forced_candidates=None,
    stats_top_k=9        
):
    """
    Disjoint Sampling with Explicit Buckets.
    1. Forced: Maintenance (Friends) & Vetting (Probation).
    2. Exploit: Discovery via Embeddings (skips Forced).
    3. Random: Pure Exploration (skips Forced & Exploit).
    """
    N = len(E_list)
    device = E_list[0].device
    E_tensor = torch.stack(E_list) 
    neigh_lists = []
    
    stats = {'purity_exploit': [], 'exploit_count': []}
    
    for i in range(N):
        # --- Bucket 1: Forced ---
        forced_set = set()
        if forced_candidates is not None and forced_candidates[i] is not None:
             forced_vals = forced_candidates[i].tolist()
             if isinstance(forced_vals, int): forced_vals = [forced_vals]
             forced_set = set(forced_vals)
        
        # --- Bucket 2: Exploit ---
        # Find closest nodes in Embedding Space
        my_emb = E_tensor[i, i].unsqueeze(0)
        all_embs = E_tensor[i] 
        dists = torch.norm(all_embs - my_emb, dim=1)
        
        _, sorted_indices_tensor = torch.sort(dists)
        sorted_indices = sorted_indices_tensor.tolist()
        
        exploit_indices = []
        # Fill budget with nodes NOT in Forced
        for candidate in sorted_indices:
            if candidate == i: continue 
            if candidate in forced_set: continue 
            
            exploit_indices.append(candidate)
            if len(exploit_indices) >= k_exploit:
                break
                
        # --- Bucket 3: Random ---
        # Exclude everyone picked so far to ensure utility
        excluded = forced_set.union(set(exploit_indices))
        excluded.add(i)
        
        remaining_indices = [idx for idx in range(N) if idx not in excluded]
        
        if len(remaining_indices) > 0:
            n_take = min(n_random, len(remaining_indices))
            random_indices = random.sample(remaining_indices, n_take)
        else:
            random_indices = []
            
        # Combine
        final_list = list(forced_set) + exploit_indices + random_indices
        combined_tensor = torch.tensor(final_list, device=device, dtype=torch.long)
        neigh_lists.append(combined_tensor)
        
        # Stats
        if node_to_cluster is not None:
            my_c = node_to_cluster[i]
            matches = [1 if node_to_cluster[n] == my_c else 0 for n in exploit_indices]
            stats['purity_exploit'].append(np.mean(matches) if matches else 0)
            stats['exploit_count'].append(len(exploit_indices))

    return neigh_lists, stats

def update_embeddings_ladder_triplet(
    E_list, 
    neighbor_ranklists_with_dists, 
    opt_list, 
    steps=10,
    alpha=2.0,  # Scale factor for dynamic margin
    trust_weights=None,  # Optional parallel structure: trust_weights[i] is a list
                          # of floats (one per evidence item) used as trust_weight.
                          # If None, trust_weight defaults to 1.0 (back-compat).
):
    N = len(E_list)
    
    for i in range(N):
        Ei = E_list[i]
        optimizer = opt_list[i]
        
        evidence = neighbor_ranklists_with_dists[i]
        if not evidence: continue

        # Resolve per-evidence trust weights, defaulting to 1.0 each
        if trust_weights is not None and i < len(trust_weights):
            tw_list = trust_weights[i]
        else:
            tw_list = [1.0] * len(evidence)
        
        for _ in range(steps):
            optimizer.zero_grad()
            total_loss = 0
            weight_sum = 0
            
            # 'evidence' is naturally ordered by Node i's physical ranks!
            # rank_idx = 0 is Node i's #1 closest neighbor
            for rank_idx, (anchor_id, neighbors, model_dists) in enumerate(evidence):
                M = neighbors.size(0)
                if M < 2: continue
                
                # --- EXTERNAL WEIGHT (Trusting the Neighbor) ---
                # v3: trust_weights from PAGANv3 (per-pair trust score).
                # v2/legacy: 1.0 for all evidence (unchanged).
                trust_weight = (float(tw_list[rank_idx])
                                  if rank_idx < len(tw_list) else 1.0)
                
                # 1. Anchored Embeddings
                center_emb = Ei[anchor_id].unsqueeze(0)
                neighbor_embs = Ei[neighbors]
                
                # 2. Embedding Distances
                d_emb = torch.norm(neighbor_embs - center_emb, p=2, dim=1)
                
                # 3. Ladder Construction
                d_closer = d_emb[:-1]
                d_farther = d_emb[1:]
                
                # 4. Dynamic Margins (With SAFE CAP)
                d_model = model_dists.to(Ei.device)
                model_gap = d_model[1:] - d_model[:-1]
                target_margin = torch.clamp(alpha * model_gap, max=5.0)
                
                # 5. Loss Calculation
                violation = d_closer - d_farther + target_margin
                raw_loss = F.relu(violation)
                
                # --- INTERNAL WEIGHT (Equal Weighting) ---
                # Every step in the ladder is equally critical, especially the Void!
                packet_loss = raw_loss.mean()
                
                total_loss += (packet_loss * trust_weight)
                weight_sum += trust_weight
                
            if weight_sum > 0:
                final_loss = total_loss / weight_sum
                final_loss.backward()
                optimizer.step()
                
    return E_list

def calc_jsd_rank_correlation(ranked_neighbors, jsd_matrix):
    """
    Computes Spearman correlation between:
    1. The rank of a neighbor based on Physical Model Distance (0, 1, 2...)
    2. The rank of a neighbor based on JSD (Ground Truth Similarity)
    
    ranked_neighbors: List[Tensor] where each tensor contains neighbor indices sorted by Model Distance.
    jsd_matrix: Tensor [N, N] of pairwise JSDs.
    """
    total_corr = 0.0
    valid_nodes = 0
    
    # Ensure JSD matrix is on CPU/Numpy for scipy
    if isinstance(jsd_matrix, torch.Tensor):
        jsd_mat = jsd_matrix.detach().cpu().numpy()
    else:
        jsd_mat = jsd_matrix

    for i, neighbors in enumerate(ranked_neighbors):
        if len(neighbors) < 3: continue # Need minimum samples for correlation
        
        # 1. Model Ranks
        # Since 'neighbors' is already sorted by Model Distance (closest first),
        # their ranks are simply 0, 1, 2, ... k
        model_ranks = np.arange(len(neighbors))
        
        # 2. JSD Ranks
        # Get the JSD values for these neighbors wrt node i
        n_indices = neighbors.cpu().numpy()
        jsd_values = jsd_mat[i, n_indices]
        
        # Rank the JSD values (Low JSD = Rank 0 = Best)
        # argsort().argsort() gives the rank of each element
        jsd_ranks = np.argsort(np.argsort(jsd_values))
        
        # 3. Correlation
        # We expect positive correlation: Low Model Rank (0) should have Low JSD Rank (0)
        corr, _ = scipy.stats.spearmanr(model_ranks, jsd_ranks)
        
        if not np.isnan(corr):
            total_corr += corr
            valid_nodes += 1
            
    return total_corr / valid_nodes if valid_nodes > 0 else 0.0

def calc_embedding_diagnostics(E_list, node_to_cluster):
    """
    Computes geometric health metrics of the embedding space.
    Handles Private Embeddings: Averaging metrics across N local views.
    """
    if not E_list: return {}
    
    # Stack: [N_observers, N_targets, Dim]
    # Example: [100, 100, 8]
    if isinstance(E_list, list):
        E = torch.stack(E_list).detach()
    else:
        E = E_list.detach()
        
    num_observers, num_targets, dim = E.shape
    device = E.device
    
    # 1. Local Spread (Variance within each node's map)
    # We want to know: "Does Node i put everyone in a pile, or spread them out?"
    # Var over targets (dim=1). Sum over feature dims (dim=2).
    # spread_per_view: [N_observers]
    spread_per_view = torch.var(E, dim=1).sum(dim=1)
    spread = spread_per_view.mean().item()
    
    # 2. Cluster Separation (Intra vs Inter Sim)
    # Compute similarity matrices for ALL views at once
    # [N, N, D] @ [N, D, N] -> [N, N, N]
    # Sim[k, i, j] = "How close Node k thinks Node i and Node j are"
    Sim = torch.bmm(E, E.transpose(1, 2))
    
    if isinstance(node_to_cluster, torch.Tensor):
        labels = node_to_cluster.to(device)
    else:
        labels = torch.tensor(node_to_cluster, device=device)
        
    # Create Ground Truth Masks (N x N)
    # label_matrix[i,j] is True if i and j share a cluster
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1) # [N, N]
    
    # Mask out self-loops (diagonal)
    mask_self = torch.eye(num_targets, device=device).bool() # [N, N]
    
    # Apply logic to all N views
    # We broaden the mask to [N, N, N] via broadcasting if needed, 
    # but since the mask is the same for all views, we can just mask the averaged Sim 
    # OR mask strictly. Let's do it per view for rigor.
    
    # Intra-Cluster Mask
    intra_mask = label_matrix & (~mask_self) # [N, N]
    
    # Inter-Cluster Mask
    inter_mask = (~label_matrix) & (~mask_self) # [N, N]
    
    # We can average Sim across observers first to get "Average Network Consensus"
    # AvgSim[i, j] = Average similarity of i and j across all N observers
    AvgSim = Sim.mean(dim=0) # [N, N]
    
    if intra_mask.sum() > 0:
        intra_sim = AvgSim[intra_mask].mean().item()
    else:
        intra_sim = 0.0
        
    if inter_mask.sum() > 0:
        inter_sim = AvgSim[inter_mask].mean().item()
    else:
        inter_sim = 0.0
        
    return {
        'emb_spread': spread,
        'sim_intra': intra_sim,
        'sim_inter': inter_sim,
        'sim_ratio': intra_sim - inter_sim # Higher is better
    }


def k_regular_clustered(num_nodes, node_to_cluster, k, device):

    neigh_lists = []
    W_local = torch.zeros((num_nodes, num_nodes), device=device)
    for node in range(num_nodes):
        cluster = node_to_cluster[node]
        pool = np.where(node_to_cluster == cluster)[0]
        pool = pool[pool != node]
        
        if k > len(pool):
            print("Too large k too small clusters, adjusting k")
            k = len(pool)
        neighbors = np.random.choice(pool, size=k, replace=False).tolist()
        neigh_lists.append(neighbors)
        W_local[node][neighbors] = 1
        W_local[node][node] = 1
        W_local[node] /= W_local[node].sum()

    return neigh_lists, W_local


def k_regular_random(num_nodes, k, device):

    neigh_lists = []
    W_local = torch.zeros((num_nodes, num_nodes), device=device)
    for node in range(num_nodes):
        pool = np.delete(np.arange(num_nodes), node)
        neighbors = np.random.choice(pool, size=k, replace=False).tolist()
        neigh_lists.append(neighbors)
        W_local[node][neighbors] = 1
        W_local[node][node] = 1
        W_local[node] /= W_local[node].sum()

    return neigh_lists, W_local

def federated_aggregation(local_models, weights):
    stack = torch.stack(local_models)
    avg = torch.sum(stack * weights.view(-1,1), dim=0)
    return avg

@torch.no_grad()
def init_identical_E_list(N: int, d: int, device, generator: torch.Generator):
    """
    Creates a single E0 [N,d], unit-normalized per row, and returns E_list of length N,
    where each entry is a clone of E0 (node-local view starts identical).
    """
    E0 = torch.randn(N, d, generator=generator, device=device) * 0.1
    # E0 = torch.nn.functional.normalize(E0, p=2, dim=1)
    # Each node owns its own copy (they will diverge later)
    return [E0.clone().requires_grad_(True) for _ in range(N)]

@torch.no_grad()
def rank_neighbors_by_model_distance(node_wts: torch.Tensor, neighbor_lists):
    """
    For each node i, compute squared Euclidean distance to each neighbor in neighbor_lists[i],
    sort ascending (closest first), and return ranked neighbors and distances.

    Returns:
      ranked_neighbors: List[LongTensor] per node
      ranked_dists:     List[Tensor(float)] per node (same lengths)
    """
    device = node_wts.device
    dtype  = node_wts.dtype
    N, D  = node_wts.shape

    ranked_neighbors = []
    ranked_dists     = []

    for i in range(N):
        neigh = neighbor_lists[i]
        if neigh.numel() == 0:
            ranked_neighbors.append(neigh)
            ranked_dists.append(torch.empty(0, dtype=dtype, device=device))
            continue

        xi = node_wts[i].unsqueeze(0)            # [1, D]
        Xn = node_wts.index_select(0, neigh)      # [k, D]
        d2 = torch.sum((Xn - xi) * (Xn - xi), dim=1)  # [k]

        vals, order = torch.sort(d2, dim=0, stable=True)
        ranked_neighbors.append(neigh[order])
        ranked_dists.append(vals)

    return ranked_neighbors, ranked_dists


@torch.no_grad()
def aggregate_neighbors(
    node_wts: torch.Tensor,            # [N, D]
    K,
    ranked_neighbors,                  # List[LongTensor], per node neighbors sorted by increasing distance
    ranked_dists=None,                 # Optional[List[Tensor]] aligned with ranked_neighbors
    weighting: str = 'inv_rank',        # 'uniform' | 'inv_dist' | 'softmax'
    return_W: bool = True,
    eps: float = 1e-8,
    # softmax params:
    temp: float = 0.8,
    self_margin: float = 0.0,
    self_w_min: float = None,
    self_w_max: float = None,
    debug_caps: bool = True,
):
    device = node_wts.device
    N, D = node_wts.shape
    W_local = torch.zeros((N, N), device=device, dtype=node_wts.dtype) if return_W else None

    for i in range(N):
        neigh = ranked_neighbors[i]           # use ALL sampled neighbors
        if neigh.numel() == 0:
            if return_W: W_local[i, i] = 1.0
            continue

        chosen = neigh                         # all neighbors
        chosen_all = torch.cat(
            [chosen, torch.tensor([i], device=device, dtype=neigh.dtype)], dim=0
        )

        if weighting == 'inv_rank':
            # [NEW] Inverse Rank Weighting (1/(r+1))
            # Rank is implicit in the order of ranked_neighbors (sorted by distance)
            # The first element is Rank 0, second is Rank 1, etc.
            k_neigh = chosen.numel()
            
            # Neighbors weights: 1/(1), 1/(2), ... 1/(k)
            ranks = torch.arange(k_neigh, device=device, dtype=node_wts.dtype)
            w_neigh = 1.0 / (ranks + 1.0)
            
            # Weight for Self:
            # We assign self the same weight as the Top-1 Neighbor (Rank 0) -> 1.0
            # This ensures we trust ourselves as much as our best friend.
            w_self = torch.tensor(1.0, device=device, dtype=node_wts.dtype)
            
            w_all = torch.cat([w_neigh, w_self.unsqueeze(0)])
            
            # Normalize to sum to 1
            w_all = w_all / (w_all.sum() + eps)
        # --- weights
        elif weighting == 'inv_dist' and ranked_dists is not None and ranked_dists[i].numel() > 0:
            d = ranked_dists[i]               # all distances to chosen
            w_neigh = 1.0 / (d + eps)
            self_w = torch.median(w_neigh) if w_neigh.numel() > 0 else torch.tensor(1.0, device=device, dtype=node_wts.dtype)
            w_all = torch.cat([w_neigh, self_w.unsqueeze(0)], dim=0)
            w_all = w_all / (w_all.sum() + eps)

        elif weighting == 'softmax' and ranked_dists is not None and ranked_dists[i].numel() > 0:
            d = ranked_dists[i].to(node_wts.dtype)
            logits_neigh = -d / max(temp, 1e-6)
            logit_self  = torch.tensor(-(self_margin / max(temp, 1e-6)), device=logits_neigh.device, dtype=logits_neigh.dtype)
            logits_all = torch.cat([logits_neigh, logit_self.view(1)], dim=0)
            w_all = torch.softmax(logits_all, dim=0)  # last index is self

            # self caps
            if w_all.numel() > 1:
                w_self = w_all[-1]
                hit = None
                if (self_w_max is not None) and (w_self > self_w_max):
                    hit = f"max({self_w_max:.3f})"
                    excess = w_self - self_w_max
                    w_all[-1] = self_w_max
                    s = w_all[:-1].sum()
                    if s > 0:
                        w_all[:-1] = w_all[:-1] + excess * (w_all[:-1] / s)
                    else:
                        w_all[:-1] = (1 - w_all[-1]) * (torch.ones_like(w_all[:-1]) / (w_all.numel() - 1))
                if (self_w_min is not None) and (w_all[-1] < self_w_min):
                    hit = f"min({self_w_min:.3f})"
                    deficit = self_w_min - w_all[-1]
                    s = w_all[:-1].sum()
                    if s > 0:
                        w_all[:-1] = w_all[:-1] - deficit * (w_all[:-1] / s)
                        w_all[-1]  = self_w_min
                        w_all[:-1] = torch.clamp(w_all[:-1], min=0.0)
                        rem = 1.0 - w_all[-1]
                        s2 = w_all[:-1].sum()
                        if s2 > 0:
                            w_all[:-1] = rem * (w_all[:-1] / s2)
                        else:
                            w_all[:-1] = rem * (torch.ones_like(w_all[:-1]) / (w_all.numel() - 1))
                    else:
                        w_all[-1] = self_w_min
                        w_all[:-1] = (1 - w_all[-1]) * (torch.ones_like(w_all[:-1]) / (w_all.numel() - 1))
                #if hit and debug_caps:
                #    print(f"[softmax-cap] node={i} hit {hit}; w_self={float(w_self):.3f} -> {float(w_all[-1]):.3f}")

        elif weighting == 'uniform':
            # uniform over neighbors + self
            w_all = torch.ones(chosen_all.numel(), device=device, dtype=node_wts.dtype)
            w_all = w_all / (w_all.sum() + eps)

        elif weighting == 'harmonic':
            inertia_ratio = 9
            w_self_val = inertia_ratio / (K + inertia_ratio)
            w_self = torch.tensor(w_self_val, device=device, dtype=node_wts.dtype)
            w_remainder = 1.0 - w_self_val
            w_neigh = torch.full((K,), w_remainder / K, device=device, dtype=node_wts.dtype)
            w_all = torch.cat([w_neigh, w_self.unsqueeze(0)])
            
        if return_W:
            W_local[i, chosen_all] = w_all
  
    updated_wts = torch.matmul(W_local, node_wts) if return_W else node_wts
    return updated_wts, W_local


@torch.no_grad()
def gather_neighbor_ranklists_with_dists(prev_ranked_neighbors, prev_ranked_dists, chosen_neighbors=None):
    """
    Like gather_neighbor_ranklists, but also returns distances.
    Output per node i: list of tuples (j, rlist_j, dists_j), where:
      - j is a chosen neighbor of i (this round's chosen, if provided; else all).
      - rlist_j is LongTensor of j's ranked neighbors from the PREVIOUS round.
      - dists_j  is Tensor of distances aligned with rlist_j (same length).
    """
    N = len(prev_ranked_neighbors)
    out = []
    for i in range(N):
        chosen = prev_ranked_neighbors[i] if chosen_neighbors is None else chosen_neighbors[i]
        if chosen.numel() == 0:
            out.append([])
            continue
        items_i = []
        for j in chosen.tolist():
            rlist_j = prev_ranked_neighbors[j]
            dlist_j = prev_ranked_dists[j] if (prev_ranked_dists is not None) else None
            if dlist_j is None:
                # keep interface consistent: empty tensor with correct device/dtype
                dlist_j = torch.empty(0, dtype=rlist_j.dtype, device=rlist_j.device).to(torch.float32)
            items_i.append((int(j), rlist_j, dlist_j))
        out.append(items_i)
    return out


def p2p_aggregation(spoke_wts, W):
    """
    W: shape [num_spokes, num_spokes]
    spoke_wts: shape [num_spokes, dim]
    returns: shape [num_spokes, dim]
    """
    return torch.mm(W, spoke_wts)