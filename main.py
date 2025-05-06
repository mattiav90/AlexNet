import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.prune as prune
import os
import math
import pandas as pd
import gc


# files
from quantizer_test import *
from fp_model import *
from qat_model import *
from quantizer_train import *


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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # load testing set
    testset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=0)

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

    # plt.show()


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


    

def compute_name(qat,bits,pruning,sparsity,loss_acc,out_name):

    _,_,acc = loss_acc[-1]
    acc_last = acc[-1]

    # format as an int. the dot create problems in the file name.
    sparsity=int(sparsity)
    bits=int(bits)
    acc_last=int(acc_last)

    name = f"qat_{qat}_bits_{bits}_pruning_{pruning}_sparsity_{sparsity}_acc_{acc_last}"
    title = f"QAT {num_bits} bits Sparsity: {int(sparsity)} on CIFAR-10"
    
    if out_name is not None:
        name=out_name
    
    path= f"./{models_path}/{name}"
    print("path: ",path)
    
    return name,title,path





# ************************************   save model  ************************************ 


# Save fp model 
def save_model_func(model, file_name, qat, out_name, stats=None, save_stats=False):

    # set the model in evaluation mode before saving it. otherwise you save values meant for training and not inference 
    # During train(): BatchNorm uses batch stats; FakeQuantize updates its observer stats.
    # During eval(): BatchNorm uses stored running stats; FakeQuantize uses frozen quantization ranges.
    model.eval()
    model.apply(torch.quantization.disable_observer)

    # calculate_sparsity_mask(model)
    # apply_pruning_mask(model)
    # make_pruning_permanent(model)

    if out_name is not None:
        file_name=out_name
        file_name=f"./{models_path}/{out_name}"


    # quantization aware training 
    if qat:

        # save the activation stats generate during training
        if save_stats:   
            if not file_name.endswith('.pth'):
                file_name += '.pth'
                
            print("Saving the model with the activation stats...  " )
            torch.save({
            'model_state_dict': model.state_dict(),
            'stats': stats
            },file_name )
        
        # save just the model, with no stats
        else:
            if not file_name.endswith('.pt'):
                file_name += '.pt'
                print("file_name: ",file_name)
                # set the model in evaluation mode before saving it. otherwise you save values meant for training and not inference 
                torch.save(model.state_dict(), file_name)
            

    # save the floating point model 
    else:

        if not file_name.endswith('.pt'):
            file_name += '.pt'
            print("file_name: ",file_name)
            # set the model in evaluation mode before saving it. otherwise you save values meant for training and not inference 
            torch.save(model.state_dict(), file_name)
            
    
    
    




# ************************************   main  ************************************ 
from config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Path to a trained model for testing")
    parser.add_argument("--out",type=str,help="Generated model output name")
    parser.add_argument("--qat",action="store_true",help="Quantization aware training activate ")
    parser.add_argument("--bit",type=int , help="quantization bits" )
    parser.add_argument("--es",type=int,help="early stopping")
    parser.add_argument("--load",type=str,help="load an already trained model")
    argom = parser.parse_args()



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
    activation_QAT_start = cfg.activation_QAT_start
    num_bits = cfg.num_bits
    symmetrical = cfg.symmetrical
    models_path = cfg.models_path
    pruning = cfg.pruning
    pruning_every = cfg.pruning_every
    pruning_ratio = cfg.pruning_ratio
    final_sparsity = cfg.final_sparsity
    lr_step_size = cfg.lr_step_size
    lr_gamma = cfg.lr_gamma
    prune_after_epoch = cfg.prune_after_epoch
    lasso_lambda = cfg.lasso_lambda
    stats_mode = cfg.stats_mode
    activation_bit = cfg.activation_bit
    
    out_name =argom.out
    
    
    if activation_bit is not None:
        print(" (!) I am forcing activation quantization to be: ",activation_bit)
    

   

    # overwriting the qauntization in the config file  
    if argom.bit is not None:
        num_bits=argom.bit
        print(f"quantizing with {num_bits}")
    
    
    # printing a beginning of computation log 
    if argom.test is None:
        print("**********************************")
        print(f"Starting training. qat: {argom.qat} bit: {num_bits} pruning: {pruning} final_sparsity: {final_sparsity} ")
        print("**********************************")
    else:
        print("**********************************")
        print("Testing a model: ", argom.test)
        print("**********************************")

	

    # load training and testing datasets
    trainset,train_loader,testset,test_loader = load_datasets()
    # create alexnet model
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = AlexNet()
    model.to(device)

    
    # ********************************************** LOAD TO TRAIN MORE **********************************************
    if argom.load is not None:
        model=load_model(argom.load,model)
        # calculate sparsity of loaded model 
        print("loading the model and checking that the sparsity is correct...")
        model = mask_frozen_weights(model)
        calculate_sparsity_mask(model)
        # apply the pruning mask to match the current sparsity. 
        model.to(cfg.device)
        print(f"Model {argom.load} loaded.")
        lr = lr * 0.5   # reduce the lr because the model is already trained
        


    # ******************************************************************************************************
    # ********************************************** TESTING **********************************************
    # ******************************************************************************************************
    if argom.test:
        
        
        # ********************************************** QAT TESTING **********************************************
        if argom.qat:
            print("testing qat.")
            model.to(device)
            model.eval()
            calculate_sparsity_zeros(model)

            # Loading a model with statistics (.pth)
            try:
                checkpoint = torch.load(argom.test,map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                stats=checkpoint['stats']
                print(" <!!!> loaded model with stats.")
            
            # load a model with no stats. (generate the stats)
            except:
                model_state_dict = torch.load(argom.test, map_location=device, weights_only=False)
                print("\n <!!!> Loaded model with NO stats. Generating activation stats... mode: ",cfg.stats_mode,"\n")
                stats = gatherStats(model,test_loader,cfg.stats_mode)
                print("Calcualted stats: ",stats,"\n\n")

            # testing the loaded model.  
            args={}
            args["log_interval"] = cfg.log_interval
            loss_temp, accuracy_temp = testQuantAware(args, model, device, test_loader, stats,
                                            act_quant=True, num_bits=num_bits, sym=cfg.symmetrical, is_test=True)




        # ********************************************** FP TESTING **********************************************
        else:
            print("testing fp precision")
            model=load_model(argom.test,model)
            model.to(cfg.device)
            calculate_sparsity_zeros(model)
            args_dict = {"log_interval": cfg.log_interval}
            test_model(args_dict, model, cfg.device, test_loader)

    
    
    
    

    # ******************************************************************************************************
    # ********************************************** TRAINING **********************************************
    # ******************************************************************************************************
    else:
        print("setting model in train mode.")
        model.train()
        
    # ********************************************** TRAINING FP **********************************************
        if not argom.qat:
            if argom.load is not None:
                print("\nFP training on model: ",argom.load,"\n")

            model, loss_acc, final_sparsity = main_train_fp(trainset,train_loader,testset,test_loader,pruning,argom.es,model=model)
            print("the final sparsity is: ", final_sparsity)
        
        
    # ********************************************** TRAINING QAT **********************************************
        elif argom.qat:
            if argom.load is not None:
                print("\nQAT training on model: ",argom.load,"\n")

            model, stats, loss_acc = main_QuantAwareTrain(trainset,train_loader,testset,test_loader,argom.out,symmetrical,pruning,argom.es,model=model)

            # make the pruning permanent before testing the last time. 
            print("\n *** finalizing pruning *** ")
            make_pruning_permanent(model)
            print("\n **** counting the zeros sparsity ****")
            calculate_sparsity_zeros(model)

            print("\n *** Testing after training *** ")
            args={}
            args["log_interval"] = cfg.log_interval  
            loss_temp, accuracy_temp = testQuantAware(args, model, device, test_loader, stats, act_quant=True, num_bits=num_bits, sym=cfg.symmetrical, is_test=True)

        # generate plot and model name
        name,title,path = compute_name(argom.qat,num_bits,pruning,final_sparsity,loss_acc,out_name)

        # plot
        plot_loss_accuracy(loss_acc, argom.qat, pruning,final_sparsity, num_bits, title=title, save_path=path)


    # ********************************************** SAVE MODEL **********************************************
        if save_model:
            if argom.qat:
                print("save model qat")
                save_model_func(model,path,True,out_name,stats=stats,save_stats=True)
            else:
                print("save model fp")
                save_model_func(model,path,out_name,False)




    # printing a beginning of computation log 
    if argom.test is None:
        print("****************************************************************")
        print(f"Starting training. qat: {argom.qat} bit: {num_bits} pruning: {pruning} final_sparsity: {final_sparsity} ")
        print("****************************************************************")
	

