import torch
import torch.nn as nn
from config import cfg
import torch.optim as optim
from quantizer import *
from pruning import *

# ************************************  train QAT  ************************************ 


def main_QuantAwareTrain(trainset,train_loader,testset,test_loader,model_name,sym,pruning,early_stopping=None, model=None):
    current_sparsity=0
    # temporary variable.
    fool=False
    if pruning and not fool:
        fool=True
    
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    model = model.to(device)
        
    model.train()
    optimizer = optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step_size,gamma=lr_gamma)  # lr_step_size=10, lr_gamma=0.1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=cfg.lr_step_size, factor=cfg.lr_gamma)


    # pruning parameters
    if pruning: 
        prune_every = cfg.pruning_every
        prune_amount = cfg.pruning_ratio

    args={}
    args["log_interval"] = cfg.log_interval
    stats={}
    loss_acc =[]
    for epoch in range(1,cfg.epochs+1):
        print(f"******* running QAT epoch {epoch} ********")
        # start QAT after some epochs number. 
        # or if the model is already quantized and has to be pruned. 
        if epoch > cfg.activation_QAT_start or pruning:
            act_quant = True
            print("Activation Quantization active")
        else:
            act_quant = False
            print("Activation Quantization non-active")

        loss_temp,accuracy_temp = trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant, current_sparsity=current_sparsity, num_bits=cfg.num_bits, sym=sym, early_stopping=early_stopping)
        loss_acc.append([epoch,loss_temp,accuracy_temp])
        # scheduler.step()
        scheduler.step(loss_temp[-1])


        if pruning and ( epoch%prune_every==0 or epoch>=prune_after_epoch ):
            print("pruning at epoch: ",epoch)
            apply_pruning(model,prune_amount)
            reset_optimizer_state(optimizer, model)

            current_sparsity = calculate_pruned_sparsity(model)
            if current_sparsity >= cfg.final_sparsity:
                pruning=False
                print(f"sparsity ({current_sparsity}) reached final: {cfg.final_sparsity}. stopping pruning.")

        # sparsity calcualation
        current_sparsity = calculate_pruned_sparsity(model,verbose=False)
        print(f"Current sparsity: {current_sparsity:.2f}")


        # print("\ncalculating one last time the accuracy....\n")
        # loss_temp, accuracy_temp = testQuantAware(
        #                 args, model, device, test_loader, stats,
        #                 act_quant=argom.qat, num_bits=num_bits, sym=symmetrical, is_test=True )
        
    # remove pruning wrappers.
    if fool:
        last_sparsity = calculate_pruned_sparsity(model)
        print("final sparsity: ", round(last_sparsity,2) )
        apply_pruning_mask(model)
        fool=False
    else: 
        last_sparsity=0

    return model, stats, loss_acc




# changed
def trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant=False, current_sparsity=0, num_bits=8, sym=False, early_stopping=None):
    model.train()

    loss_log = []
    accuracy_log = []
    acc_loss_log = []

    i = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # forward pass with fake quantization
        output, stats = quantAwareTrainingForward(model, data, stats,
                                                   num_bits=num_bits,
                                                   act_quant=act_quant,
                                                   sym=sym, mode=cfg.stats_mode)

        # calculate the loss (ON THE TRAINING SET)
        ce_loss = F.cross_entropy(output, target)


        # --- L1 regularization ---
        l1_lambda = 1e-5
        l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'weight' in name)
        loss = ce_loss + l1_lambda * l1_norm
        # -------------------------


        # replace NaN or Inf with a large finite value
        MAX_LOSS_VALUE=20
        if not torch.isfinite(loss):
            print(f"⚠️ Loss was non-finite (value: {loss.item()}). Clamping to {MAX_LOSS_VALUE}.", end='\r')
            loss = torch.tensor(MAX_LOSS_VALUE, device=loss.device, dtype=loss.dtype, requires_grad=True)



        print("training idx:", batch_idx, end='\r')
        # print("running loss backward. batch_idx: ", batch_idx)
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # make sure that the previously pruned weights are not used for training.
        # Enforce pruning masks after weight update
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_mask"):
                module.weight.data *= module.weight_mask
        

        # if current_sparsity < cfg.final_sparsity:
        #     early_stopping = 1
        # else:
        #     early_stopping = 100
                

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

            loss_log.append(loss_temp)
            accuracy_log.append(accuracy_temp)
            i += 1

            if early_stopping is not None and i >= early_stopping:
                print("Early stopping at epoch: ", epoch)
                break
        
        # clean the forward graph. to make sure it is not re-used between train and inference. 
        gc.collect()
        torch.cuda.empty_cache()

    return [loss_log, accuracy_log]




# changed
def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4, sym=False, is_test=False):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            print("testing: ",i , end='\r')
            with torch.no_grad():
                output, stats = quantAwareTrainingForward(model, data, stats,
                                                        num_bits=num_bits,
                                                        act_quant=act_quant,
                                                        sym=sym, is_test=is_test, mode=cfg.stats_mode)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set (QAT): Average loss: {:.4f}, Accuracy: {:.0f}%  ({}/{})  lr: {} '.format(
        test_loss, accuracy, correct, len(test_loader.dataset), cfg.lr ))
    
    return [test_loss, accuracy]







def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False, is_test=False, mode="minmax"):
    
    def apply_quant(tensor, name, mode="minmax"):
        if is_test:
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
            # print("passing entropy. min= ",stats[name]['entropy_min_val']," max: ",stats[name]['entropy_max_val'])
            bits_effective = cfg.activation_bit if cfg.activation_bit is not None else num_bits
            # print("quantizing with bits_effective: ",bits_effective)
            return FakeQuantOp.apply(
                tensor,
                bits_effective,
                stats[name]['entropy_min_val'],
                stats[name]['entropy_max_val'],
                sym
            )

    # -------- Layer 1 --------
    weight = FakeQuantOp.apply(model.conv1.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv1.bias, model.conv1.stride, model.conv1.padding, model.conv1.dilation, model.conv1.groups)
    x = F.relu(x)
    x = model.bn1(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1', mode=mode)
        # stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1', mode="entropy")
    if act_quant:
        x = apply_quant(x, 'conv1', mode=mode)
    x = F.max_pool2d(x, 3, 2)

    # -------- Layer 2 --------
    weight = FakeQuantOp.apply(model.conv2.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv2.bias, model.conv2.stride, model.conv2.padding, model.conv2.dilation, model.conv2.groups)
    x = F.relu(x)
    x = model.bn2(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2', mode=mode)
    if act_quant:
        x = apply_quant(x, 'conv2', mode=mode)
    x = F.max_pool2d(x, 3, 2)

    # -------- Layer 3 --------
    weight = FakeQuantOp.apply(model.conv3.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv3.bias, model.conv3.stride, model.conv3.padding, model.conv3.dilation, model.conv3.groups)
    x = F.relu(x)
    x = model.bn3(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3', mode=mode)
    if act_quant:
        x = apply_quant(x, 'conv3', mode=mode)

    # -------- Layer 4 --------
    weight = FakeQuantOp.apply(model.conv4.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv4.bias, model.conv4.stride, model.conv4.padding, model.conv4.dilation, model.conv4.groups)
    x = F.relu(x)
    x = model.bn4(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4', mode=mode)
    if act_quant:
        x = apply_quant(x, 'conv4', mode=mode)

    # -------- Layer 5 --------
    weight = FakeQuantOp.apply(model.conv5.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv5.bias, model.conv5.stride, model.conv5.padding, model.conv5.dilation, model.conv5.groups)
    x = F.relu(x)
    x = model.bn5(x)
    with torch.no_grad():
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
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1', mode=mode)
    if act_quant:
        x = apply_quant(x, 'fc1', mode=mode)

    # -------- FC2 --------
    weight = FakeQuantOp.apply(model.fc2.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc2.bias)
    x = F.relu(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2', mode=mode)
    if act_quant:
        x = apply_quant(x, 'fc2', mode=mode)

    # -------- FC3 (Output Layer) --------
    weight = FakeQuantOp.apply(model.fc3.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc3.bias)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc3', mode=mode)
    if act_quant:
        x = apply_quant(x, 'fc3', mode=mode)

    return x, stats






# ************************************   quantized inference  ************************************ 

def quantized_inference(model, x, stats, sym=False, num_bits=8):
    model.eval()
    x = x.to('cpu')
    
    scale_x, zp_x = calcScaleZeroPoint(x.min(), x.max(), num_bits)

    x, scale_next, zp_next = quantizeLayer(x, model.conv1, stats['conv1'], scale_x, zp_x, sym=sym, num_bits=num_bits)
    x = model.Maxpool(x)
    x = model.bn1(x)

    x, scale_next, zp_next = quantizeLayer(x, model.conv2, stats['conv2'], scale_next, zp_next, sym=sym, num_bits=num_bits)
    x = model.Maxpool(x)
    x = model.bn2(x)

    x, scale_next, zp_next = quantizeLayer(x, model.conv3, stats['conv3'], scale_next, zp_next, sym=sym, num_bits=num_bits)
    x = model.bn3(x)

    x, scale_next, zp_next = quantizeLayer(x, model.conv4, stats['conv4'], scale_next, zp_next, sym=sym, num_bits=num_bits)
    x = model.bn4(x)

    x, scale_next, zp_next = quantizeLayer(x, model.conv5, stats['conv5'], scale_next, zp_next, sym=sym, num_bits=num_bits)
    x = model.bn5(x)
    x = model.Maxpool(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    x = model.dropout(x)

    x, scale_next, zp_next = quantizeLayer(x, model.fc1, stats['fc1'], scale_next, zp_next, sym=sym, num_bits=num_bits)
    x = model.dropout(x)
    x, scale_next, zp_next = quantizeLayer(x, model.fc2, stats['fc2'], scale_next, zp_next, sym=sym, num_bits=num_bits)
    x, _, _ = quantizeLayer(x, model.fc3, stats['fc3'], scale_next, zp_next, sym=sym, num_bits=num_bits)

    return x

