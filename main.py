import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
import time
from config import cfg
from quantizer import *
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.utils.prune as prune
import os
import math
import pandas as pd

# ************************************  model  ************************************ 

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.ReLU = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(384)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.Maxpool(x)
        x = self.bn1(x)

        x = self.ReLU(self.conv2(x))
        x = self.Maxpool(x)
        x = self.bn2(x)

        x = self.ReLU(self.conv3(x))
        x = self.bn3(x)

        x = self.ReLU(self.conv4(x))
        x = self.bn4(x)

        x = self.ReLU(self.conv5(x))
        x = self.bn5(x)
        x = self.Maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.ReLU(self.fc1(x))
        x = self.dropout(x)
        x = self.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x



# ************************************  pruning  ************************************ 


def apply_pruning(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Apply or re-apply pruning to accumulate sparsity
            prune.l1_unstructured(module, name='weight', amount=amount)


def apply_pruning_mask(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.remove(module, 'weight')



def calculate_sparsity(model, verbose=True):
    total_zeros = 0
    total_elements = 0
    no_mask=False

    if verbose:
        print("Layer-wise sparsity:")
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Check if the weight_mask is present
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask  # This should be set after pruning
                if mask is not None:
                    num_zeros = torch.sum(mask == 0).item()
                    num_elements = mask.numel()
                    layer_sparsity = 100.0 * num_zeros / num_elements
                    
                    if verbose:
                        print(f"{name}: {layer_sparsity:.2f}% sparsity")

                    total_zeros += num_zeros
                    total_elements += num_elements
                else:
                    no_mask=True
            else:
                no_mask=True

    if no_mask:
        print(f"No weight_mask found. Sparsity calculation skipped.")
        no_mask=False
        return 0.0
        
    # Calculate the overall sparsity only if total_elements > 0
    if total_elements > 0:
        overall_sparsity = 100.0 * total_zeros / total_elements
        print(f"\nOverall sparsity: {overall_sparsity:.2f}%")
        return round(overall_sparsity,2)
    else:
        # if there is no mask on the weights yet. spardity =0.
        return 0.0



# ************************************  main_train_fp  ************************************ 



def main_train_fp(trainset,train_loader,testset,test_loader,pruning,early_stopping=None, model=None):

    # temporary variable.
    fool=False
    if pruning and not fool:
        fool=True
    
    # use or not cuda. 
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set the seed manually
    torch.manual_seed(seed)

    # creating a new AlexNet model 
    if model is None:
        model = AlexNet().to(device)
        
    # loading an already trained model
    else:
        model = model.to(device)
    
    model.train()
    optimizer= optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step_size,gamma=lr_gamma)  # lr_step_size=10, lr_gamma=0.1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=lr_step_size, factor=lr_gamma)

    
    # pruning
    if pruning:
        prune_every = cfg.pruning_every
        prune_amount = cfg.pruning_ratio

    # compose some args. 
    loss_acc = []
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1,epochs+1):
        loss_temp,acc_temp = train(args,model,device,train_loader,optimizer,epoch,test_loader,early_stopping)

        loss_acc.append([epoch,loss_temp,acc_temp])
        # scheduler.step()
        scheduler.step(loss_temp[-1])
        

        # if pruning is active. 
        if pruning and epoch % prune_every == 0: 
            print("pruning at epoch: ",epoch)
            apply_pruning(model, amount=prune_amount)
            
            current_sparsity = calculate_sparsity(model)
            if current_sparsity >= cfg.final_sparsity:
                pruning=False
                print(f"sparsity ({current_sparsity}) reached final: {cfg.final_sparsity}. stopping pruning.")

        # sparsity calcualation
        current_sparsity = calculate_sparsity(model,verbose=False)


    # remove pruning wrappers. 
    # if pruning was active during this training, it might be off by now.
    if fool:
        last_sparsity = calculate_sparsity(model)
        print("final sparsity: ", last_sparsity)
        try:
            apply_pruning_mask(model)
        except:
            # it is possible, with early stopping, that the pruning is active but has not been applied yet.
            print("no pruning was applied yet.")
        fool=False
    else:
        last_sparsity=0

    return model, loss_acc, last_sparsity





def train(args, model, device, train_loader, optimizer, epoch, test_loader,early_stopping=None):
    
    # set the model  in train mode
    loss_log = []
    accuracy_log=[]
    model.train()
    i=0
    for batch_idx, (data,target) in enumerate(train_loader):
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output= model(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        

        if batch_idx % args["log_interval"] == 0:

            print('Training.  Epoch: {} [{:.0f}% ({}/{})]\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader),
                batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))
            temp_loss,temp_accuracy = test_model(args, model, device, test_loader)

            loss_log.append(temp_loss)
            accuracy_log.append(temp_accuracy)
            i=i+1
            
            if early_stopping is not None:
                if i== early_stopping:
                    print("Early stopping at epoch: ", epoch)
                    break
    
    return loss_log, accuracy_log






# ************************************  save  model  ************************************


# Save model weights
def save_model_func(model, file_name):
    if not file_name.endswith('.pt'):
        file_name += '.pt'
    torch.save(model.state_dict(), file_name)

# Load model weights
def load_model(file_name, model):
    state_dict = torch.load(file_name, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


# ************************************  train QAT  ************************************ 


def main_QuantAwareTrain(trainset,train_loader,testset,test_loader,model_name,sym,pruning,early_stopping=None, model=None):
    
    # temporary variable.
    fool=False
    if pruning and not fool:
        fool=True
    
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


    if model is None:
        model = AlexNet().to(device)
    else:
        model = model.to(device)
        
    model.train()
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step_size,gamma=lr_gamma)  # lr_step_size=10, lr_gamma=0.1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=lr_step_size, factor=lr_gamma)


    # pruning parameters
    if pruning: 
        prune_every = cfg.pruning_every
        prune_amount = cfg.pruning_ratio

    args={}
    args["log_interval"] = log_interval
    stats={}
    loss_acc =[]
    for epoch in range(1,epochs+1):
        # start QAT after some epochs number. 
        # or if the model is already quantized and has to be pruned. 
        if epoch > start_QAT_epoch or pruning:
            act_quant = True
            print("Activation Quantization active")
        else:
            act_quant = False
            print("Activation Quantization non-active")

        loss_temp,accuracy_temp = trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant, num_bits=num_bits, sym=sym, early_stopping=early_stopping)
        loss_acc.append([epoch,loss_temp,accuracy_temp])
        # scheduler.step()
        scheduler.step(loss_temp[-1])
        
        if pruning and epoch % prune_every ==0:
            print("pruning at epoch: ",epoch)
            apply_pruning(model, amount=prune_amount)
            
            current_sparsity = calculate_sparsity(model)
            if current_sparsity >= cfg.final_sparsity:
                pruning=False
                print(f"sparsity ({current_sparsity}) reached final: {cfg.final_sparsity}. stopping pruning.")

        # sparsity calcualation
        current_sparsity = calculate_sparsity(model,verbose=False)
        print(f"Current sparsity: {current_sparsity:.2f}")
    
    # remove pruning wrappers.
    if fool:
        last_sparsity = calculate_sparsity(model)
        print("final sparsity: ", round(last_sparsity,2) )
        apply_pruning_mask(model)
        fool=False
    else: 
        last_sparsity=0
    

    return model, stats, loss_acc




# changed
def trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant=False, num_bits=8, sym=False, early_stopping=None):
    model.train()

    loss_log = []
    accuracy_log = []
    acc_loss_log = []

    i = 0

    print("******* running new QAT epoch ********")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # forward pass with fake quantization
        output, stats = quantAwareTrainingForward(model, data, stats,
                                                   num_bits=num_bits,
                                                   act_quant=act_quant,
                                                   sym=sym)

        # calculate the loss (ON THE TRAINING SET)
        loss = F.cross_entropy(output, target)

        # backward pass and update
        print("training idx:", batch_idx, end='\r')
        # in loss.backward, python uses the computation graph that was built in the forward pass, to calculte the gradients.
        # the graph is then deleted to save memory. 
        loss.backward()
        optimizer.step()

        # logging
        if batch_idx % args["log_interval"] == 0:
            print('QAT. Train Epoch: {} [{:.0f}% ({}/{})]\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), batch_idx * len(data),
                len(train_loader.dataset), loss.item()))
            
            # because testQuantAware calls quantAwareTrainingForward again, I have to detach it from the graph. 
            # calculate the loss ON THE TESTING SET.
            with torch.no_grad():
                loss_temp, accuracy_temp = testQuantAware(args, model, device, test_loader, stats,
                                                          act_quant=act_quant, num_bits=num_bits, sym=sym)

            loss_log.append(loss_temp)
            accuracy_log.append(accuracy_temp)
            i += 1

            if early_stopping is not None and i >= early_stopping:
                print("Early stopping at epoch: ", epoch)
                break

    return [loss_log, accuracy_log]




# changed
def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4, sym=False):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            print("testing: ",i , end='\r')
            output, stats = quantAwareTrainingForward(model, data, stats,
                                                      num_bits=num_bits,
                                                      act_quant=act_quant,
                                                      sym=sym)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}% ({}/{}).'.format(
        test_loss, accuracy, correct, len(test_loader.dataset)))
    
    return [test_loss, accuracy]

        

# changed
def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False):
    # -------- Layer 1 --------
    weight = FakeQuantOp.apply(model.conv1.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv1.bias, model.conv1.stride, model.conv1.padding, model.conv1.dilation, model.conv1.groups)
    x = F.relu(x)
    x = model.bn1(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'], sym)
    x = F.max_pool2d(x, 3, 2)

    # -------- Layer 2 --------
    weight = FakeQuantOp.apply(model.conv2.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv2.bias, model.conv2.stride, model.conv2.padding, model.conv2.dilation, model.conv2.groups)
    x = F.relu(x)
    x = model.bn2(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'], sym)
    x = F.max_pool2d(x, 3, 2)

    # -------- Layer 3 --------
    weight = FakeQuantOp.apply(model.conv3.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv3.bias, model.conv3.stride, model.conv3.padding, model.conv3.dilation, model.conv3.groups)
    x = F.relu(x)
    x = model.bn3(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv3']['ema_min'], stats['conv3']['ema_max'], sym)

    # -------- Layer 4 --------
    weight = FakeQuantOp.apply(model.conv4.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv4.bias, model.conv4.stride, model.conv4.padding, model.conv4.dilation, model.conv4.groups)
    x = F.relu(x)
    x = model.bn4(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv4']['ema_min'], stats['conv4']['ema_max'], sym)

    # -------- Layer 5 --------
    weight = FakeQuantOp.apply(model.conv5.weight, num_bits, None, None, sym)
    x = F.conv2d(x, weight, model.conv5.bias, model.conv5.stride, model.conv5.padding, model.conv5.dilation, model.conv5.groups)
    x = F.relu(x)
    x = model.bn5(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv5']['ema_min'], stats['conv5']['ema_max'], sym)

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
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['fc1']['ema_min'], stats['fc1']['ema_max'], sym)

    # -------- FC2 --------
    weight = FakeQuantOp.apply(model.fc2.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc2.bias)
    x = F.relu(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['fc2']['ema_min'], stats['fc2']['ema_max'], sym)

    # -------- FC3 (Output Layer) --------
    weight = FakeQuantOp.apply(model.fc3.weight, num_bits, None, None, sym)
    x = F.linear(x, weight, model.fc3.bias)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc3')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['fc3']['ema_min'], stats['fc3']['ema_max'], sym)

    return x, stats





# ************************************   test_model  ************************************ 


def test_model(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = test_loss/len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%  ({}/{})  lr: {} \n'.format(
        test_loss, accuracy, correct, len(test_loader.dataset), lr ))

    return test_loss, accuracy
    




# ************************************   load datasets  ************************************ 



# loading the datasets
def load_datasets():

    # data augmentation for training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),    # random crop with padding
        transforms.RandomHorizontalFlip(),       # randomly flip images
        transforms.Resize(input_size),           # resize to your input size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))   #usual normalization fot cifar10
    ])

    # no augmentation for test set
    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))   #usual normalization fot cifar10
    ])

    # load training set
    trainset = datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # load testing set
    testset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return trainset, train_loader, testset, test_loader



# ************************************   plot  ************************************ 


def plot_loss_accuracy(loss_acc, qat, pruning, final_sparsity, num_bits, title="Loss and Accuracy", save_path=None):
    loss_x = []
    loss_y = []
    acc_x  = []
    acc_y  = []

    epoch, loss_list, acc_list = loss_acc[-1]
    loss_last = loss_list[-1]
    acc_last = acc_list[-1]
    print("**********************************")
    print("final loss: ", round(loss_last,2))
    print("final accuracy: ", round(acc_last,2))
    print("**********************************")

    for entry in loss_acc:
        epoch, loss_list, acc_list = entry
        num_points = len(loss_list)

        # Create evenly spaced x values between current and next epoch
        x_vals = np.linspace(epoch, epoch + 1, num=num_points, endpoint=False)

        loss_x.extend(x_vals)
        loss_y.extend(loss_list)

        acc_x.extend(x_vals)
        acc_y.extend(acc_list)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(loss_x, loss_y, 'o-', color='tab:red', label='Loss', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(acc_x, acc_y, 'o-', color='tab:blue', label='Accuracy', alpha=0.7)

    plt.title(title)
    fig.tight_layout()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        
        # save a csv file. 
        csv_path = os.path.splitext(save_path)[0] + "_info.csv"
        final_loss = loss_y[-1]
        final_acc  = acc_y[-1]

        # Dummy values - replace with your actual values or arguments
        # model_name = title
        if qat:
            bits = num_bits
        else: 
            bits = "_"
        quantization = qat               # e.g., 'QAT' or 'None'
        
        sparsity = final_sparsity        # or calculate dynamically
        pruning = pruning                # or False 
        accuracy = round(final_acc, 4)
        loss = round(final_loss, 4)

        # Create a DataFrame
        df = pd.DataFrame([{
            # "Model": model_name,
            "Quant": quantization,
            "Bit": bits,
            "Prun": pruning,
            "Spars": sparsity,
            "Acc": accuracy,
            "Loss": loss
        }])

        df.to_csv(csv_path, index=False)
        print(f"CSV info saved to {csv_path}")

    plt.show()


# generate model name and plot name
def name_path(models_path,qat,loss_acc, actual_sparsity):
    
    epoch, loss_list, acc_list = loss_acc[-1]
    
    loss_last = loss_list[-1]
    acc_last = acc_list[-1]
    
    default_name="default_name"
    title="title"
    path=f"./{models_path}/{default_name}.png"
    

    # if not QAT
    if not qat:
        if pruning is True:
            default_name = f"fp_model_sp{int(actual_sparsity)}_acc{int(acc_last)}"
            title = f"FP on CIFAR-10. sparsity: {int(actual_sparsity)}"
        else:
            print("acc_last: ", acc_last)
            default_name = f"fp_model_dense_acc{int(acc_last)}"
            print("default_name: ", default_name)
            title = "FP on CIFAR-10. Dense"
        path = f"./{models_path}/{default_name}_plot.png"

    
    # if QAT
    elif qat:
        if pruning is True:
            title = f"QAT {num_bits} bits Sparsity: {int(actual_sparsity)} on CIFAR-10"
            default_name = f"qat_{num_bits}b_sp{int(actual_sparsity)}_acc{int(acc_last)}_model"
        else:
            title = f"QAT {num_bits} bits Dense on CIFAR-10"
            default_name = f"qat_{num_bits}b_dense_model_acc{int(acc_last)}"
            
        path = f"./{models_path}/{default_name}_plot.png"

    return default_name,title,path


    

def compute_name(qat,bits,pruning,sparsity,loss_acc):

    _,_,acc = loss_acc[-1]
    acc_last = acc[-1]

    # format as an int. the dot create problems in the file name.
    sparsity=int(sparsity)
    bits=int(bits)
    acc_last=int(acc_last)

    name = f"qat_{qat}_bits_{bits}_pruning_{pruning}_sparsity_{sparsity}_acc_{acc_last}"
    title = f"QAT {num_bits} bits Sparsity: {int(sparsity)} on CIFAR-10"
    
    path= f"./{models_path}/{name}"
    
    return name,title,path

# ************************************   main  ************************************ 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Path to a trained model for testing")
    parser.add_argument("--out",type=str,help="Generated model output name")
    parser.add_argument("--qat",action="store_true",help="Quantization aware training activate ")
    parser.add_argument("--bit",type=int , help="quantization bits" )
    parser.add_argument("--es",type=int,help="early stopping")
    parser.add_argument("--load",type=str,help="load an already trained model")

    args = parser.parse_args()

    # configuration file. hyperparameters
    batch_size=cfg.batch_size
    test_batch_size = cfg.test_batch_size
    epochs = cfg.epochs
    lr = cfg.lr
    momentum = cfg.momentum
    input_size = cfg.input_size
    seed = cfg.seed
    log_interval = cfg.log_interval
    save_model = cfg.save_model
    no_cuda = cfg.no_cuda
    dataset_root = cfg.dataset_root
    start_QAT_epoch = cfg.start_QAT_epoch
    num_bits = cfg.num_bits
    symmetrical = cfg.symmetrical
    models_path = cfg.models_path
    pruning = cfg.pruning
    pruning_every = cfg.pruning_every
    pruning_ratio = cfg.pruning_ratio
    final_sparsity = cfg.final_sparsity
    lr_step_size = cfg.lr_step_size
    lr_gamma = cfg.lr_gamma

   

    # overwriting the qauntization in the config file  
    if args.bit is not None:
        num_bits=args.bit
        print(f"quantizing with {num_bits}")
    
    
    # printing a beginning of computation log 
    if args.test is None:
        print("**********************************")
        print(f"Starting training. qat: {args.qat} bit: {num_bits} pruning: {pruning} final_sparsity: {final_sparsity} ")
        print("**********************************")
    else:
        print("**********************************")
        print("Testing a model: ", args.test)
        print("**********************************")

	

    # load training ant testing datasets
    trainset,train_loader,testset,test_loader = load_datasets()
    
    # load a an already trained model. to train more.
    if args.load is not None:
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = AlexNet()
        model=load_model(args.load,model)
        
        # load_sparse_model(file,model) you need to instance alexnet separately. 
        model.to(cfg.device)
        print(f"Model {args.load} loaded.")
        # decrease the learning rate, because the model is already trained.
        # this is to avoind to change the trianing rate manually. 
        lr = lr * 0.5
    else:
        model = None


    # testing a pre-trained model
    if args.test:
        # create instalce of the model structure
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = AlexNet()
        model=load_model(args.test,model)

        # model = torch.load(args.test, map_location=cfg.device)
        model.to(cfg.device)
        args_dict = {"log_interval": cfg.log_interval}
        test_model(args_dict, model, cfg.device, test_loader)

    
    # train unquantized model
    else:
        # train fp model
        if not args.qat:
            model, loss_acc, final_sparsity = main_train_fp(trainset,train_loader,testset,test_loader,pruning,args.es,model=model)
            print("the final sparsity is: ", final_sparsity)
        
        # train with QAT
        elif args.qat:
            model, stats, loss_acc = main_QuantAwareTrain(trainset,train_loader,testset,test_loader,args.out,symmetrical,pruning,args.es,model=model)



        # generate plot and model name
        name,title,path = compute_name(args.qat,num_bits,pruning,final_sparsity,loss_acc)

        # plot
        plot_loss_accuracy(loss_acc, args.qat, pruning,final_sparsity, num_bits, title=title, save_path=path)

        # save the model in compressed format. 
        if save_model:
            print("saving model...")
            save_model_func(model,path)




