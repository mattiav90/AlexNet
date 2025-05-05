import torch
import torch.nn as nn
import torch.optim as optim
from pruning import *
import torch.nn.functional as F
import copy

# files
from quantizer_train import *
from config import cfg
# ************************************  train QAT  ************************************ 


def main_QuantAwareTrain(trainset,train_loader,testset,test_loader,model_name,sym,pruning,early_stopping=None, model=None):
    current_sparsity=0
    # temporary variable.

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    torch.manual_seed(cfg.seed)

    # set the model in train mode
    model.train()
    # optimizer. stocastic gradient descent
    optimizer = optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum)
    # scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=cfg.lr_step_size, factor=cfg.lr_gamma)


    args={}
    args["log_interval"] = cfg.log_interval
    stats={}
    loss_acc =[]
    for epoch in range(1,cfg.epochs+1):

        print(f"******* running QAT epoch {epoch} ********")
        # start QAT after some epochs number. 
        # or if the model is already quantized and has to be pruned. 
        if epoch > cfg.activation_QAT_start:
            act_quant = True
            print("Activation Quantization active")
        else:
            act_quant = False
            print("Activation Quantization non-active")

        # each call is 1 epoch run. 
        loss_temp,accuracy_temp = trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant, current_sparsity=current_sparsity, num_bits=cfg.num_bits, sym=sym, early_stopping=early_stopping)
        loss_acc.append([epoch,loss_temp,accuracy_temp])
        
        # the learning rate scheduler uses loss to calibrate. 
        scheduler.step(loss_temp[-1])

        # make sure that it is not training the pruned weights. (if there are any)
        calculate_pruned_sparsity(model)


    return model, stats, loss_acc




# changed
def trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant=False, current_sparsity=0, num_bits=8, sym=False, early_stopping=None):
    model.train()
    loss_log = []
    accuracy_log = []
    i = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print("training idx:", batch_idx, end='\r')
        optimizer.zero_grad()
        
        # forward pass with fake quantization
        output, stats = quantAwareTrainingForward(model, data, stats,
                                                   num_bits=num_bits,
                                                   act_quant=act_quant,
                                                   sym=sym, mode=cfg.stats_mode)

        # calculate the loss (ON THE TRAINING SET)
        loss = F.cross_entropy(output, target)

        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        
        # logging
        if batch_idx % args["log_interval"] == 0:
            print('QAT. Train Epoch: {} [{:.0f}% ({}/{})] Loss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), batch_idx * len(data),
                len(train_loader.dataset), loss.item()))
            
            # because testQuantAware calls quantAwareTrainingForward again, I have to detach it from the graph. 
            # calculate the loss ON THE TESTING SET.
            with torch.no_grad():
                
                loss_temp, accuracy_temp = testQuantAware(args, model, device, test_loader, stats,
                                                          act_quant=act_quant, num_bits=num_bits, sym=sym, is_test=True)

            # the cumulated loss is from the testing set. 
            loss_log.append(loss_temp)
            accuracy_log.append(accuracy_temp)
           
            # early stopping for debug 
            i += 1
            if early_stopping is not None and i >= early_stopping:
                print("Early stopping at epoch: ", epoch)
                break
        

    return [loss_log, accuracy_log]




# quantization aware testing. 
def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4, sym=False, is_test=False):

    model.eval()
    test_loss = 0
    correct = 0

    # do not buils a gradient graph. 
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            print("testing: ",i , end='\r')

            output = quantAwareTestingForward(model, data, stats,
                                                    num_bits=num_bits,
                                                    act_quant=act_quant,
                                                    sym=sym, mode=cfg.stats_mode)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set (QAT): Average loss: {:.4f}, Accuracy: {:.0f}%  ({}/{})  lr: {} '.format(
        test_loss, accuracy, correct, len(test_loader.dataset), cfg.lr ))
    
    return [test_loss, accuracy]





# forward pass during training. 
# apply fake quantization to weights and act
# update the activation statistics. 
# FakeQuantOp.apply by default applies minmax quantization
# mode parameter only effects the activations quantization mode
def quantAwareTrainingForward(model, x, stats, sym=False, num_bits=8, act_quant=False, mode="minmax"):
    
    def apply_quant(tensor, name, mode="minmax"):
        if mode=="minmax":
            return FakeQuantOp.apply(
                tensor,
                num_bits,
                stats[name]['min_val'],
                stats[name]['max_val'],
                sym
            )
        elif mode=="entropy":
            return FakeQuantOp.apply(
                tensor,
                num_bits,
                stats[name]['entropy_min_val'],
                stats[name]['entropy_max_val'],
                sym
            )

    # -------- Layer 1 --------
    weight = FakeQuantOp.apply(model.conv1.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv1.bias, model.conv1.stride, model.conv1.padding, model.conv1.dilation, model.conv1.groups)
    x = F.relu(x)
    x = model.bn1(x)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1', mode=mode)
    if act_quant:
        x = apply_quant(x, 'conv1', mode=mode)
    x = F.max_pool2d(x, 3, 2)

    # -------- Layer 2 --------
    weight = FakeQuantOp.apply(model.conv2.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv2.bias, model.conv2.stride, model.conv2.padding, model.conv2.dilation, model.conv2.groups)
    x = F.relu(x)
    x = model.bn2(x)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2', mode=mode)
    if act_quant:
        x = apply_quant(x, 'conv2', mode=mode)
    x = F.max_pool2d(x, 3, 2)

    # -------- Layer 3 --------
    weight = FakeQuantOp.apply(model.conv3.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv3.bias, model.conv3.stride, model.conv3.padding, model.conv3.dilation, model.conv3.groups)
    x = F.relu(x)
    x = model.bn3(x)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3', mode=mode)
    if act_quant:
        x = apply_quant(x, 'conv3', mode=mode)

    # -------- Layer 4 --------
    weight = FakeQuantOp.apply(model.conv4.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv4.bias, model.conv4.stride, model.conv4.padding, model.conv4.dilation, model.conv4.groups)
    x = F.relu(x)
    x = model.bn4(x)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4', mode=mode)
    if act_quant:
        x = apply_quant(x, 'conv4', mode=mode)

    # -------- Layer 5 --------
    weight = FakeQuantOp.apply(model.conv5.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv5.bias, model.conv5.stride, model.conv5.padding, model.conv5.dilation, model.conv5.groups)
    x = F.relu(x)
    x = model.bn5(x)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5', mode=mode)
    if act_quant:
        x = apply_quant(x, 'conv5', mode=mode)

    # -------- Pooling + Flatten --------
    x = F.max_pool2d(x, 3, 2)
    x = F.adaptive_avg_pool2d(x, (6, 6))
    x = torch.flatten(x, 1)
    x = model.dropout(x)

    # -------- FC1 --------
    weight = FakeQuantOp.apply(model.fc1.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc1.bias)
    x = F.relu(x)
    x = model.dropout(x)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1', mode=mode)
    if act_quant:
        x = apply_quant(x, 'fc1', mode=mode)

    # -------- FC2 --------
    weight = FakeQuantOp.apply(model.fc2.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc2.bias)
    x = F.relu(x)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2', mode=mode)
    if act_quant:
        x = apply_quant(x, 'fc2', mode=mode)

    # -------- FC3 (Output Layer) --------
    weight = FakeQuantOp.apply(model.fc3.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc3.bias)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc3', mode=mode)
    if act_quant:
        x = apply_quant(x, 'fc3', mode=mode)

    return x, stats





# forward pass during testing. 
# apply quantization on both activations and weights. 
# do not update the statistics. 
def quantAwareTestingForward(model, x, stats, sym=False, num_bits=8, act_quant=False, mode="minmax"):
    model.eval()

    def apply_quant(tensor, name, mode="minmax"):
        tensor = tensor.detach()
        
        if mode=="minmax":
            return FakeQuantOp.apply(
                tensor,
                num_bits,
                stats[name]['min_val'],
                stats[name]['max_val'],
                sym
            )
        elif mode=="entropy":
            return FakeQuantOp.apply(
                tensor,
                num_bits,
                stats[name]['entropy_min_val'],
                stats[name]['entropy_max_val'],
                sym
            )

    # -------- Layer 1 --------
    weight = FakeQuantOp.apply(model.conv1.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv1.bias, model.conv1.stride, model.conv1.padding, model.conv1.dilation, model.conv1.groups)
    x = F.relu(x)
    x = model.bn1(x)
    if act_quant:
        x = apply_quant(x, 'conv1', mode=mode)
    x = F.max_pool2d(x, 3, 2)

    # -------- Layer 2 --------
    weight = FakeQuantOp.apply(model.conv2.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv2.bias, model.conv2.stride, model.conv2.padding, model.conv2.dilation, model.conv2.groups)
    x = F.relu(x)
    x = model.bn2(x)
    if act_quant:
        x = apply_quant(x, 'conv2', mode=mode)
    x = F.max_pool2d(x, 3, 2)

    # -------- Layer 3 --------
    weight = FakeQuantOp.apply(model.conv3.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv3.bias, model.conv3.stride, model.conv3.padding, model.conv3.dilation, model.conv3.groups)
    x = F.relu(x)
    x = model.bn3(x)
    if act_quant:
        x = apply_quant(x, 'conv3', mode=mode)

    # -------- Layer 4 --------
    weight = FakeQuantOp.apply(model.conv4.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv4.bias, model.conv4.stride, model.conv4.padding, model.conv4.dilation, model.conv4.groups)
    x = F.relu(x)
    x = model.bn4(x)
    if act_quant:
        x = apply_quant(x, 'conv4', mode=mode)

    # -------- Layer 5 --------
    weight = FakeQuantOp.apply(model.conv5.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv5.bias, model.conv5.stride, model.conv5.padding, model.conv5.dilation, model.conv5.groups)
    x = F.relu(x)
    x = model.bn5(x)
    if act_quant:
        x = apply_quant(x, 'conv5', mode=mode)

    # -------- Pooling + Flatten --------
    x = F.max_pool2d(x, 3, 2)
    x = F.adaptive_avg_pool2d(x, (6, 6))
    x = torch.flatten(x, 1)
    
    # -------- FC1 --------
    weight = FakeQuantOp.apply(model.fc1.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc1.bias)
    x = F.relu(x)
    if act_quant:
        x = apply_quant(x, 'fc1', mode=mode)

    # -------- FC2 --------
    weight = FakeQuantOp.apply(model.fc2.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc2.bias)
    x = F.relu(x)
    if act_quant:
        x = apply_quant(x, 'fc2', mode=mode)

    # -------- FC3 (Output Layer) --------
    weight = FakeQuantOp.apply(model.fc3.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc3.bias)
    if act_quant:
        x = apply_quant(x, 'fc3', mode=mode)

    return x

