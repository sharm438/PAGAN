import torch
import utils as utils
import models as models

# Global variables that persist in the worker process
_global_test_loader = None
_device = None
_net_name = None
_dims = None
_batch_size = None

def init_worker(dataset_name, net_name, inp_dim, out_dim, batch_size, gpu_id, fraction):
    """
    Initializes the worker process.
    Loads the global test set once so it doesn't need to be reloaded for every node.
    """
    global _global_test_loader, _device, _net_name, _dims, _batch_size
    
    # 1. Setup Device
    # We use the specified GPU if available, otherwise CPU.
    # Note: All workers in the pool will share this logic. 
    # If gpu_id >= 0, they will all try to use that GPU.
    if gpu_id >= 0 and torch.cuda.is_available():
        _device = torch.device(f"cuda:{gpu_id}")
    else:
        _device = torch.device("cpu")
        
    _net_name = net_name
    _dims = (inp_dim, out_dim)
    _batch_size = batch_size
    
    # 2. Load Global Test Data
    # We use the existing utils loader. We set lr=0.0 as it's not used for loading.
    # This might print data loading logs for every worker, which is expected.
    # We assume 'utils' refers to the module compatible with the main script.
    print(f"[Worker] Initializing on {_device}...")
    train_obj = utils.load_data(dataset_name, batch_size, lr=0.0, fraction=fraction)
    _global_test_loader = train_obj.test_data

def evaluate_node(args):
    """
    Evaluates a single node's model.
    args: (node_weights_cpu, local_test_pair_cpu_or_none)
    
    Returns: (global_loss, global_acc, local_loss, local_acc)
    """
    wts, local_pair = args
    
    # 1. Move weights to the worker's compute device
    wts = wts.to(_device)
    
    # 2. Reconstruct Model
    inp_dim, out_dim = _dims
    model = utils.vec_to_model(wts, _net_name, inp_dim, out_dim, _device)
    
    # 3. Global Evaluation
    # evaluate_global_metrics puts data batches to device automatically
    g_loss, g_acc = utils.evaluate_global_metrics(model, _global_test_loader, _device)
    
    # 4. Local Evaluation (if data provided)
    l_loss, l_acc = float('nan'), float('nan')
    if local_pair is not None:
        X, y = local_pair
        # Create a temp loader for this node's local test set
        # X, y are expected to be on CPU (from main process) or shared memory
        loader = utils.make_loader_from_tensors(X, y, _batch_size, shuffle=False)
        l_loss, l_acc = utils.evaluate_global_metrics(model, loader, _device)
        
    # Return floats to ensure no graph attachments survive
    return float(g_loss), float(g_acc), float(l_loss), float(l_acc)