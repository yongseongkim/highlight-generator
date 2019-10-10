import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # For add x to F(x), reduce size of x
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        first_ch = 32
        self.conv = nn.Conv2d(3, first_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(first_ch)
        l1_out_ch = 32
        self.layer1 = self.make_layer(first_ch, l1_out_ch, block, num_blocks[0], stride=1)
        l2_out_ch = 64
        self.layer2 = self.make_layer(l1_out_ch, l2_out_ch, block, num_blocks[1], stride=2)
        l3_out_ch = 128
        self.layer3 = self.make_layer(l2_out_ch, l3_out_ch, block, num_blocks[2], stride=2)
        l4_out_ch = 256
        self.layer4 = self.make_layer(l3_out_ch, l4_out_ch, block, num_blocks[3], stride=2)
        self.linear = nn.Linear(l4_out_ch * 16 * 9, num_classes)

    def make_layer(self, in_channels, out_channels, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        prev_out = in_channels
        for stride in strides:
            layers.append(block(prev_out, out_channels, stride))
            prev_out = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)  # flatten
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(ResidualBlock, [3, 4, 6, 3])


def lr_scheduler(epoch):
    lr = 1e-3
    if epoch > 160:
        lr *= 0.5e-3
    elif epoch > 140:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


def predict(model, loss_fun, testloader):
    accuracy, test_loss = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fun(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            accuracy = len(c[c == True]) / len(c)
    return accuracy, test_loss


def train_model(model, loss_fun, optimizer, train_loader, testloader, epochs=50):
    steps = len(train_loader)
    train_loss, test_loss, test_accuracy = [], [], []
    for epoch in range(epochs):
        lpe = 0  # loss per epoch

        # control learning rate
        for g in optimizer.param_groups:
            g['lr'] = lr_scheduler(epoch)

        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fun(outputs, labels)

            optimizer.zero_grad()
            lpe += loss.item()
            loss.backward()
            optimizer.step()
        lpe /= steps
        train_loss.append(lpe)
        predicted_accuracy, predicted_loss = predict(
            model, loss_fun, testloader)
        test_accuracy.append(predicted_accuracy)
        test_loss.append(predicted_loss)
        print('Epoch [{}/{}], Loss: {:.4f} Val Accuracy: {:.4f}'.format(epoch + 1, epochs, lpe, predicted_accuracy))
    return model, train_loss, test_loss, test_accuracy


if __name__ == "__main__":
    batch_size = 64
    dataset = dset.ImageFolder(root="../videos/raw_files/ingame-dataset/",
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(dataset_size)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18().to(device)
    model, train_loss, test_loss, test_accuracy = train_model(model,
                                                              nn.CrossEntropyLoss(),
                                                              torch.optim.Adam(
                                                                  model.parameters(), lr=0.001),
                                                              train_loader,
                                                              test_loader,
                                                              epochs=200)
    # torch.save(model.state_dict(), os.path.join(base_dir, 'cifar10_resnet18_naive.ckpt'))
