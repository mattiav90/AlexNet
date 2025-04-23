import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
import argparse

from model import AlexNet 
from conf import conf

def plot_metrics(loss_history, accuracy_history):
    iterations = list(range(1, len(loss_history) + 1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iterations, loss_history, label="Loss", color='blue')
    plt.title("Training Loss")
    plt.xlabel("Iteration (per 100 steps)")
    plt.ylabel("Loss")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(iterations, accuracy_history, label="Accuracy", color='green')
    plt.title("Validation Accuracy")
    plt.xlabel("Iteration (per 100 steps)")
    plt.ylabel("Accuracy (%)")
    plt.grid()

    plt.tight_layout()
    plt.show()





def main(model_load, output_name, qat, epoc):
    
    # Hyperparameters
    num_epochs = conf.num_epochs
    train_batch_size =conf.train_batch_size
    test_batch_size = conf.test_batch_size
    learning_rate = conf.learning_rate
    momentum = conf.momentum
    weight_decay = conf.weight_decay
    device = conf.device
    cifar10_dataset = conf.cifar10_dataset




    # Test set transformation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    testset = torchvision.datasets.CIFAR10(root=cifar10_dataset, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    # Train set transformation with augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    trainset = torchvision.datasets.CIFAR10(root=cifar10_dataset, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    # Define AlexNet model with QAT
    model = AlexNet(num_classes=10, qat=qat).to(device)

    if qat:
        print("QAT activated")
        model.fuse_model()  # Fuse conv and relu layers
        torch.quantization.prepare_qat(model, inplace=True)  # Prepare for QAT

    # Loading an already trained model
    if model_load:
        state_dict = torch.load(model_load, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f" ****** Loading the model: {model_load}. ****** ")
    else:
        print(" ****** The model will be trained from scratch. ****** ")

    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    accuracy_history = []
    loss_history = []

    print(f"Starting the training. train_batch_size: {train_batch_size} learning_rate: {learning_rate} ")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        model.train()
        for i, (inputs, labels) in enumerate(tqdm.tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                avg_loss = running_loss / 100
                running_accuracy = model.evaluate(testloader, device)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[{epoch + 1}, {i + 1}] learning_rate: {current_lr:.5f} loss: {avg_loss:.3f} accuracy: {running_accuracy:.3f}")
                loss_history.append(avg_loss)
                accuracy_history.append(running_accuracy)
                running_loss = 0.0
        
        scheduler.step()

    print("Finished Training")

    if not output_name:
        output_name = "new_model.pth"

    # Save the model and convert it to real quantized values
    if qat:
        model.cpu()
        model.eval()
        # Convert the model to a real quantized model
        quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        model_to_save = quantized_model
    else:
        model_to_save = model

    torch.save(model_to_save.state_dict(), output_name)

    return loss_history, accuracy_history








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load",type=str,help="load a model and train starting on that.")
    parser.add_argument("--out",type=str,help="give a name of choise to the output model.")
    parser.add_argument("--qat",action="store_true",help="QAT")
    parser.add_argument("--epoc",type=int,help="how many epocs for training.")
    args = parser.parse_args()
    
    # arguments
    model_load=args.load
    output_name=args.out
    qat=args.qat
    epoc=args.epoc

    loss_history, accuracy_history = main(model_load,output_name,qat,epoc)
    # Call the plot function after training finishes
    plot_metrics(loss_history, accuracy_history)

    
