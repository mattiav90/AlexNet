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



def calculate_sparsity(model):
    total_zeros = 0
    total_elements = 0

    print("Layer-wise sparsity:")
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight_mask'):
                weight = module.weight
                mask = module.weight_mask  # This exists only if pruning has been applied

                num_zeros = torch.sum(mask == 0).item()
                num_elements = mask.numel()
                layer_sparsity = 100.0 * num_zeros / num_elements

                print(f"{name}: {layer_sparsity:.2f}% sparsity")

                total_zeros += num_zeros
                total_elements += num_elements

    overall_sparsity = 100.0 * total_zeros / total_elements
    overlall_sparsity = round(overall_sparsity, 2)
    print(f"\nOverall sparsity: {overall_sparsity:.2f}%")

    return overall_sparsity


# ************************************  main_train_fp  ************************************ 

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




def main_train_fp(trainset,train_loader,testset,test_loader,pruning,early_stopping=None, model=None):


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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step_size,gamma=lr_gamma)
    
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
        test_model(args,model,device,test_loader)
        loss_acc.append([epoch,loss_temp,acc_temp])

        # if pruning is active. 
        if pruning and epoch % prune_every == 0: 
            print("pruning at epoch: ",epoch)
            apply_pruning(model, amount=prune_amount)
            
            current_sparsity = calculate_sparsity(model)
            print(f"Current sparsity: {current_sparsity:.4f}")
            if current_sparsity >= cfg.final_sparsity:
                pruning=False
                print(f"sparsity ({current_sparsity}) reached final: {cfg.final_sparsity}. stopping pruning.")

    

    # remove pruning wrappers
    if pruning:
        last_sparsity = calculate_sparsity(model)
        apply_pruning_mask(model)
    else:
        last_sparsity=0

    return model, loss_acc, last_sparsity



# ************************************  train QAT  ************************************ 


def main_QuantAwareTrain(trainset,train_loader,testset,test_loader,model_name,symm,early_stopping=None, model=None):
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step_size,gamma=lr_gamma)
    args={}
    args["log_interval"] = log_interval

    stats={}
    loss_acc =[]
    
    for epoch in range(1,epochs+1):
        # do not start immediately to do qat, only after some iterations
        if epoch > start_QAT_epoch:
            act_quant = True
        else:
            act_quant = False

        loss_temp,accuracy_temp = trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant, num_bits=num_bits, sym=symm, early_stopping=early_stopping)
        scheduler.step()

        loss_acc.append([epoch,loss_temp,accuracy_temp])
    

    return model, stats, loss_acc






def trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant=False, num_bits=8 , sym=False, early_stopping=None):
    model.train()

    loss_log = []
    accuracy_log = []
    acc_loss_log=[]

    i=0

    for batch_idx, (data,target) in enumerate(train_loader):
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # forward pass with fake quantization. 
        output, conv1weight, conv2weight, conv3weight, conv4weight,conv5weight,\
        fc1weight, fc2weight, fc3weight, stats= quantAwareTrainingForward(model, data, stats,
                                                                   num_bits=num_bits,
                                                                   act_quant=act_quant,
                                                                   sym=sym)
        
        # for the backward pass, the weights have to be restored to the original non quantized values.
        # Fake quantization is only applied during the forward pass (to mimic quantized inference).
        model.conv1.weight.data = conv1weight
        model.conv2.weight.data = conv2weight
        model.conv3.weight.data = conv3weight
        model.conv4.weight.data = conv4weight
        model.conv5.weight.data = conv5weight
        model.fc1.weight.data   = fc1weight
        model.fc2.weight.data   = fc2weight
        model.fc3.weight.data   = fc3weight

        # calculate the loss
        loss = F.cross_entropy(output,target)
        # do the backward pass
        loss.backward()
        # update the weights
        optimizer.step()

        # log the loss 

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{:.0f}% ({}/{})]\tLoss: {:.6f}'.format(
            epoch, 100. * batch_idx / len(train_loader) ,batch_idx * len(data),
            len(train_loader.dataset), loss.item()))
            # test the accuracy with quantized weights.
            loss_temp, accuracy_temp = testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits, sym=sym)

            loss_log.append(loss_temp)
            accuracy_log.append(accuracy_temp)
            i=i+1

            if early_stopping is not None:
                if i== early_stopping:
                    print("Early stopping at epoch: ", epoch)
                    break

    return [loss_log, accuracy_log]



# testing the prediction using fake quantization. 
def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4, sym=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            output, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, \
            fc1weight, fc2weight, fc3weight, stats = quantAwareTrainingForward(model, data, stats,
                                                                    num_bits=num_bits,
                                                                    act_quant=act_quant,
                                                                    sym=sym)

            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.conv3.weight.data = conv3weight
            model.conv4.weight.data = conv4weight
            model.conv5.weight.data = conv5weight
            model.fc1.weight.data = fc1weight
            model.fc2.weight.data = fc2weight
            model.fc3.weight.data = fc3weight

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}% ({}/{}). lr: {} \n'.format(
        test_loss, accuracy, 
        correct, len(test_loader.dataset), lr ))
    
    return [test_loss,accuracy]
        
        



def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False):


    #  ********** layer 1 **********
    # the original weightrs are saved on the side. 
    conv1weight = model.conv1.weight.data
    # apply fake quantization to the weights. 
    model.conv1.weight.data = FakeQuantOp.apply(model.conv1.weight.data, num_bits,  None, None, sym)
    # use the fake quantized weights to perform the convolution.  
    x = F.relu(model.conv1(x))
    x = model.bn1(x)
    # keep track of statistics on the activations, to calculate quantization factors later
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0],-1),stats,'conv1')
    # when quantizing the acitivations, use the runtime statistics
    if act_quant:
        x = FakeQuantOp.apply(x,num_bits,stats['conv1']['ema_min'],stats['conv1']['ema_max'], sym) 
    x = F.max_pool2d(x,3,2)


    #  ********** layer 2 **********
    conv2weight = model.conv2.weight.data
    model.conv2.weight.data = FakeQuantOp.apply(model.conv2.weight.data, num_bits,  None, None, sym)
    x=F.relu(model.conv2(x))
    x=model.bn2(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'], sym) 
    x = F.max_pool2d(x, 3, 2)


    #  ********** layer 3 **********
    conv3weight = model.conv3.weight.data
    model.conv3.weight.data = FakeQuantOp.apply(model.conv3.weight.data, num_bits,  None, None, sym)
    x = F.relu(model.conv3(x))
    x = model.bn3(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv3']['ema_min'], stats['conv3']['ema_max'], sym) 


    #  ********** layer 4 **********
    conv4weight = model.conv4.weight.data
    model.conv4.weight.data = FakeQuantOp.apply(model.conv4.weight.data, num_bits, None, None, sym)
    x = F.relu(model.conv4(x))
    x = model.bn4(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv4']['ema_min'], stats['conv4']['ema_max'], sym) 


    #  ********** layer 5 **********
    conv5weight = model.conv5.weight.data
    model.conv5.weight.data = FakeQuantOp.apply(model.conv5.weight.data, num_bits,  None, None, sym)
    x = F.relu(model.conv5(x))
    x = model.bn5(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv5']['ema_min'], stats['conv5']['ema_max'], sym) 


    #  ********** CONV to FC **********
    x = F.max_pool2d(x, 3, 2)
    x = F.adaptive_avg_pool2d(x,(6, 6))
    x = torch.flatten(x, 1)
    x = model.dropout(x)

    #  ********** layer 6. FC **********
    fc1weight = model.fc1.weight.data
    model.fc1.weight.data = FakeQuantOp.apply(model.fc1.weight.data, num_bits, None, None, sym)
    x = F.relu(model.fc1(x))
    x = model.dropout(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['fc1']['ema_min'], stats['fc1']['ema_max'], sym) 


    #  ********** layer 7. FC **********
    fc2weight = model.fc2.weight.data
    model.fc2.weight.data = FakeQuantOp.apply(model.fc2.weight.data, num_bits,  None, None, sym)
    x = F.relu(model.fc2(x))
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['fc2']['ema_min'], stats['fc2']['ema_max'], sym) 


    #  ********** layer 8. FC **********
    fc3weight = model.fc3.weight.data
    model.fc3.weight.data = FakeQuantOp.apply(model.fc3.weight.data, num_bits,  None, None, sym)
    x = model.fc3(x)
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc3')
    if act_quant:
        x = FakeQuantOp.apply(x,num_bits,stats['fc3']['ema_min'], stats['fc3']['ema_max'], sym) 
    

    return x, conv1weight, conv2weight, conv3weight, \
                conv4weight,conv5weight,fc1weight, fc2weight, fc3weight, stats






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


def plot_loss_accuracy(loss_acc, title="Loss and Accuracy", save_path=None):
    loss_x = []
    loss_y = []
    acc_x  = []
    acc_y  = []
    

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
        
        info_path = os.path.splitext(save_path)[0] + "_info.txt"
        final_loss = loss_y[-1]
        final_acc  = acc_y[-1]
        
        with open(info_path, 'w') as f:
            f.write(f"Title: {title}\n")
            f.write(f"Final Loss: {final_loss:.4f}\n")
            f.write(f"Final Accuracy: {final_acc:.4f}\n")
        
        print(f"Info file saved to {info_path}")

    plt.show()


# generate model name and plot name
def name_path(models_path,qat,loss_acc, actual_sparsity):
    
    epoch, loss_list, acc_list = loss_acc[-1]
    
    loss_last = loss_list[-1]
    acc_last = acc_list[-1]
    

    # if not QAT
    if not qat:
        if pruning is True:
            default_name = f"fp_model_sp{int(actual_sparsity)}_acc{int(acc_last)}"
            title = f"FP on CIFAR-10. sparsity: {int(actual_sparsity)}"
        else:
            default_name = f"fp_model_dense_acc{int(acc_last)}"
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


# save the model
def save_model_func(model,model_name,out_name=None):
    # plot accuracy, loss and save the model

    if out_name:
        model_name=f"./{models_path}/{out_name}.pth"
    else:
        model_name=f"./{models_path}/{default_name}.pth"

    torch.save(model,model_name)

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
        model = torch.load(args.load, map_location=cfg.device)
        model.to(cfg.device)
        print(f"Model {args.load} loaded.")
        # decrease the learning rate, because the model is already trained.
        # this is to avoind to change the trianing rate manually. 
        lr = lr * 0.5
    else:
        model = None


    # testing a pre-trained model
    if args.test:
        model = torch.load(args.test, map_location=cfg.device)
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
            model, stats, loss_acc = main_QuantAwareTrain(trainset,train_loader,testset,test_loader,args.out,symmetrical,args.es,model=model)
  

        # generate plot and model name
        default_name,title,path = name_path(models_path,args.qat, loss_acc, final_sparsity)

        # plot
        plot_loss_accuracy(loss_acc, title=title, save_path=path)

        # save the model
        if save_model:
            save_model_func(model,default_name,args.out)




