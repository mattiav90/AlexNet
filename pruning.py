import torch
import torch.nn as nn
import config as cfg
import torch.nn.utils.prune as prune

# ************************************************************************************************************
# ************************************************************************************************************
# ************************************************************************************************************
# ************************************  pruning  ************************************ 

# apply pruning. 
def apply_pruning(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Apply or re-apply pruning to accumulate sparsity
            prune.l1_unstructured(module, name='weight', amount=amount)


# calcualted pruning sparsity 
def calculate_pruned_sparsity(model, verbose=True):
    total_zeros = 0
    total_elements = 0
    found_mask = False

    if verbose:
        print("Layer-wise sparsity (via mask):")

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                num_elements = mask.numel()
                num_zeros = (mask == 0).sum().item()
                total_elements += num_elements
                total_zeros += num_zeros
                found_mask = True

                if verbose:
                    sparsity = 100.0 * num_zeros / num_elements
                    print(f"{name}: {sparsity:.2f}% sparsity")

    if not found_mask:
        if verbose:
            print("No pruning masks found. Returning 0% sparsity.")
        return 0.0

    overall_sparsity = 100.0 * total_zeros / total_elements
    if verbose:
        print(f"(*) Overall sparsity: {overall_sparsity:.2f}%")

    return round(overall_sparsity, 2)




# just calculate the weigghts sparsity, do not apply the mask
def calculate_weight_sparsity(model, verbose=True):
    total_zeros = 0
    total_elements = 0
    found_weights = False
    
    print("Layer-wise sparsity (counting zero weights):")

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight'):
                weight = module.weight.data
                num_elements = weight.numel()
                num_zeros = (weight == 0).sum().item()
                total_elements += num_elements
                total_zeros += num_zeros
                found_weights = True

                if verbose:
                    sparsity = 100.0 * num_zeros / num_elements
                    print(f"{name}: {sparsity:.2f}% sparsity")

    if not found_weights:
        if verbose:
            print("No weights found. Returning 0% sparsity.")
        return 0.0

    overall_sparsity = 100.0 * total_zeros / total_elements
    if verbose:
        print(f"(*) Overall sparsity: {overall_sparsity:.2f}%")

    return round(overall_sparsity, 2)




# permanently apply the pruning mask 
def apply_pruning_mask(model):
    print("removing the pruning mask. inside function log.")
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.remove(module, 'weight')



def reset_optimizer_state(optimizer, model):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'momentum_buffer' in state:
                state['momentum_buffer'].zero_()



def make_pruning_permanent(model):
    print("Making pruning permanent (zeroed weights, no mask).")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask') and hasattr(module, 'weight_orig'):
                # Force mask application: in-place
                with torch.no_grad():
                    module.weight_orig *= module.weight_mask

                # Now remove pruning (will copy masked weight into .weight)
                prune.remove(module, 'weight')




    
def apply_dummy_pruning(model):
    # Apply dummy pruning to match saved state_dict keys
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.identity(module, 'weight')  # adds `weight_orig` and `weight_mask`







# reapply pruning mask to loaded model 
def mask_frozen_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Only apply mask to weights that exist and are trainable
            if hasattr(module, 'weight') and module.weight is not None:
                # Build binary mask: 1 where weight != 0, 0 where weight == 0
                weight_mask = (module.weight != 0).to(dtype=torch.float32)

                # Apply custom mask (prune)
                prune.custom_from_mask(module, name='weight', mask=weight_mask)
    return model
