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
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1,epochs+1):
        train(args,model,device,train_loader,optimizer,epoch,test_loader)
        test_model(args,model,device,test_loader)

    if save_model:
        if out_name:
            model_name=f"{out_name}.pth"
        else:
            model_name="new_model.pth"
        
        torch.save(model,model_name)
    
    return model



# ************************************  train QAT  ************************************ 


def main_QuantAwareTrain(trainset,train_loader,testset,test_loader):
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
    for epoch in range(1,epochs+1):
        # do not start immediately to do qat, only after some iterations
        if epoch > start_QAT_epoch:
            act_quant = True
        else:
            act_quant = False

        stats = trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant, num_bits=num_bits)
        testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

    if (save_model):
        torch.save(model, "new_model_qat.pth")

    return model, stats



def trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant=False, num_bits=8 ):
    model.train()

    for batch_idx, (data,target) in enumerate(train_loader):
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        conv1weight, conv2weight, conv3weight, conv4weight,conv5weight,\
        fc1weight, fc2weight, stats= quantAwareTrainingForward(model, data, stats,
                                                                   num_bits=num_bits,
                                                                   act_quant=act_quant)





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

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%  ({}/{}) \n'.format(
        test_loss, 100. * correct / len(test_loader.dataset), correct, len(test_loader.dataset) ))




# ************************************   load datasets  ************************************ 



# loading the datasets
def load_datasets():
    global batch_size,test_batch_size,epochs,lr,momentum,input_size,seed,log_interval,save_model,no_cuda,dataset_root

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







# ************************************   main  ************************************ 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Path to a trained model for testing")
    parser.add_argument("--out",type=str,help="generated model output name")
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
        model = main_train_fp(trainset,train_loader,testset,test_loader,args.out)
