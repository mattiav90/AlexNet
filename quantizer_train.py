import torch
import torch.nn.functional as F
from collections import namedtuple

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point




def quantize_tensor(x, num_bits=8, min_val=None, max_val=None, sym=False):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    # asymmetrical quantization
    if not sym:
        scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
        q_x = zero_point + x / scale
        qmin = 0.
        qmax = 2. ** num_bits - 1.
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()

    # symmetrical quantization
    if sym:
        # if a min and a max are passed. that is the calculation for activation quantization threshould.
        # I should implement that also. here.
        max_val = max(abs(min_val),abs(max_val))
        scale = max_val / (2**(num_bits-1)-1)
        zero_point=0
        q_x = x/scale
        qmax = 2. ** num_bits - 1.
        q_x.clamp_( -qmax , qmax ).round_()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)




def dequantize_tensor(q_x):
    # print("dequantize_tensor. zxero_point: ", q_x.zero_point)
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)



# during statistics update, make sure that the graph is disconnected from the data. 
# use .item() and .cpu() for that. 
def updateStats(x, stats, key, mode="minmax"):
    """
    Update activation statistics using min max-based metrics.

    Args:
        x: Activation tensor.
        stats: Dictionary tracking per-layer stats.
        key: Identifier for layer.
    """
    if mode=="minmax":
        x = x.detach()

        max_val = torch.max(x, dim=1)[0].cpu()
        min_val = torch.min(x, dim=1)[0].cpu()

        # create a new key
        if key not in stats:
            stats[key] = {  "max":   max_val.sum().item(),
                            "min":   min_val.sum().item(),
                            "total": 1
                        }
        
        # update the present key
        else:
            stats[key]['max']   += max_val.sum().item()
            stats[key]['min']   += min_val.sum().item()
            stats[key]['total'] += 1

        weighting = 2.0 / (stats[key]['total']) + 1

        if 'ema_min' in stats[key]:
            stats[key]['ema_min'] = weighting * (min_val.mean().item()) + (1 - weighting) * stats[key]['ema_min']
        else:
            stats[key]['ema_min'] = weighting * (min_val.mean().item())

        if 'ema_max' in stats[key]:
            stats[key]['ema_max'] = weighting * (max_val.mean().item()) + (1 - weighting) * stats[key]['ema_max']
        else:
            stats[key]['ema_max'] = weighting * (max_val.mean().item())

        stats[key]['min_val'] = stats[key]['min'] / stats[key]['total']
        stats[key]['max_val'] = stats[key]['max'] / stats[key]['total']
        
        # print("\n\nmin max. min: ", stats[key]['min_val'], " max: ", stats[key]['max_val'])
    
    elif mode=="entropy":
        """
        Update activation statistics using entropy-based metrics.

        Args:
            x: Activation tensor.
            stats: Dictionary tracking per-layer stats.
            key: Identifier for layer.
            num_bins: Number of bins for histogram.
            eps: Small value to avoid log(0).
        """
        num_bins=128
        eps=1e-8
        
        # Detach and flatten
        x_flat = x.detach().view(-1).cpu()
        
        # Compute min and max from actual data
        x_min = x_flat.min().item()
        x_max = x_flat.max().item()

        if x_min == x_max:
            # Constant activation — no entropy
            entropy = 0.0
            entropy_min_val = x_min
            entropy_max_val = x_max
        else:
            # Histogram over [min, max]
            hist = torch.histc(x_flat, bins=num_bins, min=x_min, max=x_max)
            prob = hist / (hist.sum() + eps)
            
            # Entropy
            entropy = -(prob * (prob + eps).log()).sum().item()

            # Use cumulative distribution to find where the entropy "mass" is
            cdf = torch.cumsum(prob, dim=0)
            nonzero_min = (cdf >= 0.01).nonzero(as_tuple=True)[0]
            nonzero_max = (cdf <= 0.99).nonzero(as_tuple=True)[0]

            if len(nonzero_min) == 0 or len(nonzero_max) == 0:
                entropy_min_idx = 0
                entropy_max_idx = num_bins - 1
            else:
                entropy_min_idx = nonzero_min.min().item()
                entropy_max_idx = nonzero_max.max().item()

            # Convert bin indices back to values
            bin_width = (x_max - x_min) / num_bins
            entropy_min_val = x_min + entropy_min_idx * bin_width
            entropy_max_val = x_min + entropy_max_idx * bin_width

        # Initialize stats
        if key not in stats:
            stats[key] = {
                "total": 1,
                "entropy_sum": entropy,
                "ema_entropy": entropy,
                "entropy_min_val": entropy_min_val,
                "entropy_max_val": entropy_max_val,
            }
            
        # update stats
        else:
            stats[key]['total'] += 1
            stats[key]['entropy_sum'] += entropy
            w = 2.0 / stats[key]['total'] + 1
            stats[key]['ema_entropy'] = w * entropy + (1 - w) * stats[key]['ema_entropy']
            stats[key]['entropy_min_val'] = w * entropy_min_val + (1 - w) * stats[key]['entropy_min_val']
            stats[key]['entropy_max_val'] = w * entropy_max_val + (1 - w) * stats[key]['entropy_max_val']
            

        stats[key]['entropy_avg'] = stats[key]['entropy_sum'] / stats[key]['total']

        # print("\n\nentropy. min: ", stats[key]['entropy_min_val'], " max: ", stats[key]['entropy_max_val'])
        
    return stats





# # custom pytorch autograd function. 
# # custom forward and backward pass can be defined by using the @staticmethod decorator.
class FakeQuantOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None, sym=False):
        # snap the weights to quantized rapresentation
        x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val, sym=sym)
        # print("FakeQuantOp forward. x: ", x[0][0][0])
        # convert it back to floating point to allow training. 
        x = dequantize_tensor(x)
        return x

    # 
    @staticmethod
    def backward(ctx, grad_output,  num_bits=8, min_val=None, max_val=None, sym=False):
        # Apply the straight-through estimator (STE)
        # Pass the gradient through unchanged.
        return grad_output, None, None, None, None




