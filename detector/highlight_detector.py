import torch
import torch.nn as nn
import torch.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import copy
import os
import time
from models import HighlightDetector
from dataset import HighlightDataset, ToTensor


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50):
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            for images, scores in dataloaders[phase]:
                images = images.to(device)
                scores = scores.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    outputs = outputs.view(outputs.size(0))
                    loss = criterion(outputs, scores)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model


batch_size = 8
num_epochs = 10
len_sequence = 7


if __name__ == "__main__":
    curdirpath = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = HighlightDataset(
        len_sequence=len_sequence,
        highlight_dir=os.path.join(
            curdirpath, './dataset/highlight'
        ),
        non_highlight_dir=os.path.join(
            curdirpath, './dataset/non-highlight'
        ),
        transform=transforms.Compose([
            ToTensor()
        ]))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.seed(dataset_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    dataloaders = {
        'train': torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices)),
        'val': torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    }

    print('Dataset Size: {}, Train Dataset Size: {}, Val Dataset Size: {}'.format(dataset_size, len(train_indices), len(val_indices)))
    model = HighlightDetector(features_dim=64,
                            hidden_dim=32,
                            layer_dim=1,
                            resnet_pretrained=True)
    model_ft, hist = train_model(model,
                                dataloaders,
                                nn.MSELoss(),
                                torch.optim.Adam(model.parameters(), lr=0.001),
                                num_epochs=num_epochs)
