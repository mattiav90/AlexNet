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



# ************************************  main_train_fp  ************************************ 

def train(args, model, device, train_loader, optimizer, epoch, test_loader):
    
    # set the model  in train mode
    model.train()
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
            test_model(args, model, device, test_loader)




def main_train_fp(trainset,train_loader,testset,test_loader,out_name):


    # use or not cuda. 
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set the seed manually
    torch.manual_seed(seed)

    # instance of AlexNet
    model = AlexNet().to(device)
    optimizer= optim.SGD(model.parameters(),lr=lr,momentum=momentum)

    # compose some args. 
    loss_temp = []
    acc_temp = []
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1,epochs+1):
        train(args,model,device,train_loader,optimizer,epoch,test_loader)
        loss_temp,acc_temp=test_model(args,model,device,test_loader)

        loss_acc.append([epoch,loss_temp,acc_temp])

    if save_model:
        if out_name:
            model_name=f"{out_name}.pth"
        else:
            model_name="new_model.pth"
        
        torch.save(model,model_name)
    
    return model, loss_acc



# ************************************  train QAT  ************************************ 


def main_QuantAwareTrain(trainset,train_loader,testset,test_loader,model_name):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
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

        loss_temp,accuracy_temp = trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant, num_bits=num_bits)
        scheduler.step()

        loss_acc.append([epoch,loss_temp,accuracy_temp])

    if (save_model):
        if model_name:
            model_name=f"{model_name}.pth"
        else:
            model_name="new_model_qat.pth"

        torch.save(model, model_name)

    return model, stats, loss_acc






def trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant=False, num_bits=8 ):
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
                                                                   sym=True)
        
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
            loss_temp, accuracy_temp = testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

            loss_log.append(loss_temp)
            accuracy_log.append(accuracy_temp)
            i=i+1

            # if i==3:
            #     break

    return [loss_log, accuracy_log]



# testing the prediction using fake quantization. 
def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4):
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
                                                                    sym=True)

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

    print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}% ({}/{}) \n'.format(
        test_loss, accuracy, 
        correct, len(test_loader.dataset) ))
    
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%  ({}/{}) \n'.format(
        test_loss, accuracy, correct, len(test_loader.dataset) ))
    
    return [test_loss,accuracy]




# ************************************   load datasets  ************************************ 



# loading the datasets
def load_datasets():

    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ]
    )
    # load training set
    trainset= datasets.CIFAR10(root=dataset_root,train=True,download=True,transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
    # load testing set
    testset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return trainset,train_loader,testset,test_loader



# ************************************   plot  ************************************ 


def plot_loss_accuracy(loss_acc, title="Loss and Accuracy", save_path=None):
    loss_x = []
    loss_y = []
    acc_x = []
    acc_y = []

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

    plt.show()

# ************************************   main  ************************************ 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Path to a trained model for testing")
    parser.add_argument("--out",type=str,help="Generated model output name")
    parser.add_argument("--qat",action="store_true",help="Quantization aware training activate ")
    args = parser.parse_args()

    # hyperparameters
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




    # load training and testing datasets
    trainset,train_loader,testset,test_loader = load_datasets()

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
            model, loss_acc = main_train_fp(trainset,train_loader,testset,test_loader,args.out)
            title = "FP on CIFAR-10"
            path = "fp_plot.png"
        
        # train with QAT
        elif args.qat:
            model, stats, loss_acc = main_QuantAwareTrain(trainset,train_loader,testset,test_loader,args.out)
            title = "QAT on CIFAR-10"
            path = "qat_plot.png"

        
        plot_loss_accuracy(loss_acc, title=title, save_path=path)







    # epochs = range(1, len(loss_list) + 1)

    # plt.figure(figsize=(10, 5))
    
    # # Plot Loss
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, loss_list, label='Loss', color='red', marker='o')
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss over Epochs")
    # plt.grid(True)

    # # Plot Accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, accuracy_list, label='Accuracy', color='blue', marker='o')
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy over Epochs")
    # plt.grid(True)

    # plt.suptitle(title)
    # plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path)
    # else:
    #     plt.show()