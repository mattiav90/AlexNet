# functions to train fp model

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



def train_fp(trainset,train_loader,testset,test_loader):


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
        torch.save(model,"new_model.pth")
    
    return model




