
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def view_classification(image, probabilities):
    probabilities = probabilities.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    image = image.permute(1, 2, 0)
    denormalized_image= image / 2 + 0.5
    ax1.imshow(denormalized_image)
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def main():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,shuffle=False)

    print('Number of images in the training dataset:', len(train_set))
    print('Number of images in the testing dataset:', len(test_set))

    print(f"Shape of the images in the training dataset: {train_loader.dataset[0][0].shape}")

    fig, axes = plt.subplots(1, 10, figsize=(12, 3))
    for i in range(10):
        image = train_loader.dataset[i][0].permute(1, 2, 0)
        denormalized_image = image / 2 + 0.5
        axes[i].imshow(denormalized_image)
        axes[i].set_title(classes[train_loader.dataset[i][1]])
        axes[i].axis('off')
    plt.show()

    class ConvNeuralNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = nn.Conv2d(3, 64, 3)
            self.conv2 = nn.Conv2d(64, 128, 3)

            self.pool = nn.MaxPool2d(2, stride=2)

            self.fc1 = nn.Linear(128 * 6 * 6, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.log_softmax(self.fc3(x), dim=1)
            return x

    net = ConvNeuralNet()
    net.to(device)

    summary(net, (3, 32, 32))

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}/{epochs}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    images, _ = next(iter(test_loader))

    image = images[3]
    batched_image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        log_probabilities = net(batched_image)

    probabilities = torch.exp(log_probabilities).squeeze().cpu()
    view_classification(image, probabilities)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    torch.save(net.state_dict(), "./image_model")

if __name__ == "__main__":
    main()
