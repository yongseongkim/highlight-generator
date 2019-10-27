import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import cv2
import copy
import os
import sys
import time

curdirpath = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


class HighlightDetector(nn.Module):

    def __init__(self, features_dim, sequence_dim, hidden_dim, layer_dim, resnet_pretrained):
        super(HighlightDetector, self).__init__()
        self.features_dim = features_dim
        self.sequence_dim = sequence_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # 7장 중 하이라이트 장면에 포함되는 만큼 점수를 준다. ex) 10장 중 5장이 하이라이트 장면이면 output = 5
        self.lstm = nn.LSTM(features_dim, hidden_dim, layer_dim)
        self.resnet = models.resnet18(pretrained=resnet_pretrained)
        self.resnet.fc = nn.Linear(512, features_dim)
        self.h2o = nn.Linear(hidden_dim, 1)

    def forward(self, imgs):
        batch_size, timesteps, C, H, W = imgs.size()
        resnet_in = imgs.view(batch_size * timesteps, C, H, W)
        resnet_out = self.resnet(resnet_in)
        features = resnet_out.view(batch_size, timesteps, -1)
        out, (hn, cn) = self.lstm(features)
        out = self.h2o(out[:, -1, :])
        return out


class HighlightDataset(Dataset):
    def __init__(self, len_sequence, highlight_dir, non_highlight_dir, transform=None):
        self.simg_width = 16 * 14
        self.len_sequence = len_sequence
        self.highlight_dir = highlight_dir
        self.non_highlight_dir = non_highlight_dir
        self.hightlight_paths = [os.path.join(
            highlight_dir, f) for f in os.listdir(self.highlight_dir)]
        self.non_hightlight_paths = [os.path.join(
            non_highlight_dir, f) for f in os.listdir(self.non_highlight_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.hightlight_paths) + len(self.non_hightlight_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= len(self.hightlight_paths):
            idx = idx - len(self.hightlight_paths)
            img = cv2.imread(self.non_hightlight_paths[idx])
            score = 0
        else:
            img = cv2.imread(self.hightlight_paths[idx])
            score = 7
        imgs = np.asarray([img[:,
                               i * self.simg_width: (i+1) * self.simg_width,
                               :] for i in range(self.len_sequence)])
        sample = imgs, score
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        imgs, score = sample[0], sample[1]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        transpoed_imgs = np.asarray([img.transpose((2, 0, 1)) for img in imgs])
        return torch.from_numpy(transpoed_imgs).float(), torch.tensor(score, dtype=torch.float)


batch_size = 8
num_epochs = 10
len_sequence = 7
dataset = HighlightDataset(
    len_sequence=len_sequence,
    highlight_dir=os.path.join(
        curdirpath, '../videos/raw_files/highlight_dataset/highlight'),
    non_highlight_dir=os.path.join(
        curdirpath, '../videos/raw_files/highlight_dataset/non-highlight'),
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
print('Dataset Size: {}, Train Dataset Size: {}, Val Dataset Size: {}'.format(
    dataset_size, len(train_indices), len(val_indices)))

model = HighlightDetector(features_dim=64,
                          sequence_dim=len_sequence,
                          hidden_dim=32,
                          layer_dim=1,
                          resnet_pretrained=True)
model_ft, hist = train_model(model,
                             dataloaders,
                             nn.MSELoss(),
                             torch.optim.Adam(model.parameters(), lr=0.001),
                             num_epochs=num_epochs)
