import torch
import torch.nn.functional as F
from collections import namedtuple

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def visualise(x, axs):
    x = x.view(-1).cpu().numpy()
    axs.hist(x)

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


def calcScaleZeroPointSym(min_val, max_val, num_bits=8):
    # Calc Scale
    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    return scale, 0




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




def quantize_tensor_sym(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    q_x = x / scale

    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale=scale, zero_point=0)




def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())



def updateStats(x, stats, key, mode="minmax"):
    
    if mode=="minmax":
        max_val, _ = torch.max(x, dim=1)
        min_val, _ = torch.min(x, dim=1)


        if key not in stats:
            stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
        else:
            stats[key]['max'] += max_val.sum().item()
            stats[key]['min'] += min_val.sum().item()
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
            entropy_min_idx = (cdf >= 0.01).nonzero(as_tuple=True)[0].min().item()
            entropy_max_idx = (cdf <= 0.99).nonzero(as_tuple=True)[0].max().item()

            # Convert bin indices back to values
            bin_width = (x_max - x_min) / num_bins
            entropy_min_val = x_min + entropy_min_idx * bin_width
            entropy_max_val = x_min + entropy_max_idx * bin_width

        # Initialize or update stats
        if key not in stats:
            stats[key] = {
                "total": 1,
                "entropy_sum": entropy,
                "ema_entropy": entropy,
                "entropy_min_val": entropy_min_val,
                "entropy_max_val": entropy_max_val,
            }
            
        else:
            stats[key]['total'] += 1
            stats[key]['entropy_sum'] += entropy
            w = 2.0 / stats[key]['total'] + 1
            stats[key]['ema_entropy'] = w * entropy + (1 - w) * stats[key]['ema_entropy']
            stats[key]['entropy_min_val'] = w * entropy_min_val + (1 - w) * stats[key]['entropy_min_val']
            stats[key]['entropy_max_val'] = w * entropy_max_val + (1 - w) * stats[key]['entropy_max_val']
            
            # print("key: ",key," is in the stats")

        stats[key]['entropy_avg'] = stats[key]['entropy_sum'] / stats[key]['total']

        # print("\n\nentropy. min: ", stats[key]['entropy_min_val'], " max: ", stats[key]['entropy_max_val'])
        
    return stats





# custom pytorch autograd function. 
# custom forward and backward pass can be defined by using the @staticmethod decorator.
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

# test
# x = torch.tensor([1, 2, 3, 4]).float()
# print(FakeQuantOp.apply(x))




def quantize_tensor_sym(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    q_x = x / scale

    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale=scale, zero_point=0)


def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())


def quantizeLayer(x, layer, stat, scale_x, zp_x, vis=False, axs=None, X=None, y=None, sym=False, num_bits=8):

    W = layer.weight.data
    B = layer.bias.data

    if sym:
        w = quantize_tensor_sym(layer.weight.data, num_bits=num_bits)
        # print("quantized tensor w: ",w[0][0][0])
        b = quantize_tensor_sym(layer.bias.data, num_bits=num_bits)
        # print("quantized tensor b: ",b[0])
    else:
        w = quantize_tensor(layer.weight.data, num_bits=num_bits)
        b = quantize_tensor(layer.bias.data, num_bits=num_bits)

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    if vis:
        axs[X, y].set_xlabel("Visualising weights of layer: ")
        visualise(layer.weight.data, axs[X, y])

    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    if sym:
        try:
            scale_next, zero_point_next = calcScaleZeroPointSym(min_val=stat['entropy_min_val'], max_val=stat['entropy_max_val'])
        except:
            scale_next, zero_point_next = calcScaleZeroPointSym(min_val=stat['min'], max_val=stat['max'])
    else:
        try:
            scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['entropy_min_val'], max_val=stat['entropy_max_val'])
        except:
            scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])


    if sym:
        X = x.float()
        layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data)
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data)
    else:
        X = x.float() - zp_x
        layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data - zp_w)
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data + zp_b)

    if sym:
        x = (layer(X))
    else:
        x = (layer(X)) + zero_point_next

    x.round_()


    x = F.leaky_relu(x)

    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next






