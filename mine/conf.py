import torch

class conf:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    train_batch_size = 128
    test_batch_size = 100
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    cifar10_dataset = "./data"

