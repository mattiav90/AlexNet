
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from pruning import *

from config import cfg
# ************************************************************************************************************
# ************************************************************************************************************
# ************************************************************************************************************
# ************************************  main_train_fp  ************************************ 


def main_train_fp(trainset,train_loader,testset,test_loader,pruning,early_stopping=None, model=None):

    # temporary variable.
    fool=False
    if pruning and not fool:
        fool=True
    
    # use or not cuda. 
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set the seed manually
    torch.manual_seed(cfg.seed)
        
    # loading an already trained model
    model = model.to(device)
    
    model.train()
    optimizer= optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step_size,gamma=lr_gamma)  # lr_step_size=10, lr_gamma=0.1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=cfg.lr_step_size, factor=cfg.lr_gamma)

    
    # pruning
    if pruning:
        prune_every = cfg.pruning_every
        prune_amount = cfg.pruning_ratio

    # compose some args. 
    loss_acc = []
    args = {}
    args["log_interval"] = cfg.log_interval
    for epoch in range(1,cfg.epochs+1):
        
        print("******** training fp model on a new epoch: ",epoch," ********")
        loss_temp,acc_temp = train(args,model,device,train_loader,optimizer,epoch,test_loader,early_stopping)

        loss_acc.append([epoch,loss_temp,acc_temp])
        # scheduler.step()
        scheduler.step(loss_temp[-1])
        

        # if pruning is active. 
        if pruning and ( epoch % prune_every==0 ) : 
            print("pruning at epoch: ",epoch)
            apply_pruning(model,prune_amount)
            
            
            # comment ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    if hasattr(module, 'weight_mask'):
                        print(f"{name} pruning mask applied: {module.weight_mask.shape}")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            current_sparsity = calculate_sparsity_mask(model)
            if current_sparsity >= cfg.final_sparsity:
                pruning=False
                print(f"sparsity ({current_sparsity}) reached final: {cfg.final_sparsity}. stopping pruning.")

        # sparsity calcualation
        current_sparsity = calculate_sparsity_mask(model,verbose=False)


    # remove pruning wrappers. 
    # if pruning was active during this training, it might be off by now.
    if fool:
        last_sparsity = calculate_sparsity_mask(model)
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
        
        print("training idx:", batch_idx, end='\r')
        
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output= model(data)
        loss = F.cross_entropy(output,target)
        
        # ++++++++++ weight regularization ++++++++++
        # Cross-entropy loss.
        # base_loss = F.cross_entropy(output, target)
        # # L1 penalty over all learnable parameters
        # l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)
        # # Total loss = base loss + λ * L1_penalty
        # # to keep the weights small. 
        # loss = base_loss + lasso_lambda * l1_penalty
        # +++++++++++++++++++++++++++++++++++++++++++
        
        # compute the gradient in the backward pass
        loss.backward() 
        # update the weights
        optimizer.step()
        
        
        if batch_idx % args["log_interval"] == 0:

            print('Training.  Epoch: {} [{:.0f}% ({}/{})]\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader),
                batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))
            with torch.no_grad():  # detach the graph
                temp_loss,temp_accuracy = test_model(args, model, device, test_loader)

            loss_log.append(temp_loss)
            accuracy_log.append(temp_accuracy)
            i=i+1
            
            if early_stopping is not None:
                if i== early_stopping:
                    print("Early stopping at epoch: ", epoch)
                    break
    
    return loss_log, accuracy_log





# # Load model weights
# def load_model(file_name, model):
#     try:
#         state_dict = torch.load(file_name, map_location='cpu')
#         model.load_state_dict(state_dict)
#     except:
#         apply_dummy_pruning(model)
#         state_dict = torch.load(file_name, map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model





def test_model(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            print("testing idx:", i, end='\r')

    test_loss = test_loss/len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}%  ({}/{})  lr: {} '.format(
        test_loss, accuracy, correct, len(test_loader.dataset), cfg.lr ))

    return test_loss, accuracy
    

