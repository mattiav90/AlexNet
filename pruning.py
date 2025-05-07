import torch
import torch.nn as nn
import config as cfg
import torch.nn.utils.prune as prune

# ************************************************************************************************************
# ************************************************************************************************************
# ************************************************************************************************************
# ************************************  pruning  ************************************ 

# prune the network
def prune_model(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Apply or re-apply pruning to accumulate sparsity
            prune.l1_unstructured(module, name='weight', amount=amount)




# calcualted pruning sparsity. use the mask 
def calculate_sparsity_mask(model, verbose=True):
    total_zeros = 0
    total_elements = 0
    found_mask = False
    print("calculating sparsity via MASK")

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
            print("(*) No pruning masks found. Returning 0% sparsity.")
        return 0.0

    overall_sparsity = 100.0 * total_zeros / total_elements
    if verbose:
        print(f"(*) Overall sparsity: {overall_sparsity:.2f}%")

    return round(overall_sparsity, 2)




# just calculate the weigghts sparsity, do not apply the mask
def calculate_sparsity_zeros(model, verbose=True, eps=0.0):
    total_zeros = 0
    total_elements = 0
    found_weights = False
    print("calculating sparsity via ZEROS")


    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                num_elements = weight.numel()
                if eps > 0.0:
                    num_zeros = (weight.abs() < eps).sum().item()
                else:
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
def finalize_pruning(model):
    print("removing the pruning mask. inside function log.")
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.remove(module, 'weight')





# make pruning permanent (zero out the weights that are masked out) to save the model 
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




#  load a pruned model and apply the mask to avoid training the weights. 
def mask_frozen_weights(model):
    print("Reapplying pruning masks to freeze zeroed weights...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                # Create a binary mask: 1 where weight != 0, 0 where weight == 0
                with torch.no_grad():
                    weight_mask = (module.weight != 0).float()

                # Apply the pruning mask using PyTorch pruning utilities
                prune.custom_from_mask(module, name='weight', mask=weight_mask)

                # Register backward hook to zero out gradients at pruned locations
                def hook_factory(mask_tensor):
                    def hook(grad):
                        return grad * mask_tensor
                    return hook

                module.weight.register_hook(hook_factory(weight_mask))

                # Make sure .weight_orig requires grad (and is a leaf tensor)
                if hasattr(module, 'weight_orig'):
                    module.weight_orig.requires_grad_(True)

    return model
