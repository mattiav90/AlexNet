import torch
import torch.nn.functional as F
from collections import namedtuple
from quantizer_train import updateStats

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


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
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


def quantizeLayer(x, layer, stat, scale_x, zp_x, vis=False, axs=None, X=None, y=None, sym=False, num_bits=8):

    W = layer.weight.data
    B = layer.bias.data

    if sym:
        w = quantize_tensor_sym(layer.weight.data, num_bits=num_bits)
        b = quantize_tensor_sym(layer.bias.data, num_bits=num_bits)
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
        scale_next, zero_point_next = calcScaleZeroPointSym(min_val=stat['min'], max_val=stat['max'])
    else:
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






def gatherActivationStats(model, x, stats, mode):
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1', mode )

    x = F.relu(model.conv1(x))
    x = model.bn1(x)

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2', mode )
    x = F.relu(model.conv2(x))
    x = model.bn2(x)
    x = F.max_pool2d(x, 3, 2)

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3', mode )
    x = F.relu(model.conv3(x))
    x = model.bn3(x)

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4', mode )
    x = F.relu(model.conv4(x))
    x = model.bn4(x)

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5', mode )
    x = F.relu(model.conv5(x))
    x = model.bn5(x)

    x = F.max_pool2d(x, 3, 2)
    x = F.adaptive_avg_pool2d(x,(6,6))
    x = torch.flatten(x, 1)
    print("x.shape: ",x.shape)
    # x = x.view(-1, 1250) #CIFAR10

    stats = updateStats(x, stats, 'fc1', mode )
    x = F.relu(model.fc1(x))

    stats = updateStats(x, stats, 'fc2', mode )
    x = F.relu(model.fc2(x))

    stats = updateStats(x, stats, 'fc3', mode )
    x = model.fc3(x)

    return stats




def gatherStats(model, test_loader, mode):
    device = 'cpu'
    model.eval()
    stats = {}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats, mode)

    final_stats = {}
    for key, value in stats.items():
        if mode == "minmax":
            final_stats[key] = {
                "min_val": value["min"] / value["total"],
                "max_val": value["max"] / value["total"],
                "ema_min": value["ema_min"],
                "ema_max": value["ema_max"]
            }
        elif mode == "entropy":
            final_stats[key] = {
                "entropy_avg": value["entropy_sum"] / value["total"],
                "ema_entropy": value["ema_entropy"],
                "entropy_min_val": value["entropy_min_val"],
                "entropy_max_val": value["entropy_max_val"]
            }

    return final_stats



