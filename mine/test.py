import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import matplotlib.pyplot as plt

from model import AlexNet  # Ensure this has the fused + prepared QAT version

def test_model(model_path):
    device = torch.device("cpu")  # Quantized models must run on CPU

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # Create model and fuse layers
    model_fp32 = AlexNet(num_classes=10)
    model_fp32.fuse_model()  # Ensure this method exists in your model class

    # Prepare for quantization
    model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model_fp32, inplace=True)

    # Convert to quantized version
    model = torch.quantization.convert(model_fp32.eval(), inplace=False)

    # Load quantized state dict
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"Quantized model loaded from: {model_path}")
    else:
        print(f"Model file not found at: {model_path}")
        return

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the quantized model on the CIFAR-10 test images: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, required=True,
                        help="Path to the quantized model state_dict")
    args = parser.parse_args()

    test_model(args.m)
