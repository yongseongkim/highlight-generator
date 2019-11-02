import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import copy
import cv2
import os
import sys
import time
import matplotlib.pyplot as plt
from models import HighlightDetector
from dataset import HighlightDataset, ToTensor


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50):
    since = time.time()
    hist = {
        'train': [],
        'val': []
    }
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
            hist[phase].append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, hist


def load_dataset(len_sequence, batch_size, val_ratio=0.1):
    dataset = HighlightDataset(
        len_sequence=len_sequence,
        highlight_dir=os.path.join(
            curdirpath, './dataset/highlight_seq' + str(len_sequence)
        ),
        non_highlight_dir=os.path.join(
            curdirpath, './dataset/non-highlight_seq' + str(len_sequence)
        ),
        transform=transforms.Compose([
            ToTensor()
        ]))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    np.random.seed(dataset_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print('Dataset Size: {}, Train Dataset Size: {}, Val Dataset Size: {}'.format(
        dataset_size, len(train_indices), len(val_indices)))
    dataloaders = {
        'train': torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices)),
        'val': torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    }
    return dataloaders


if __name__ == "__main__":
    curdirpath = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) < 2:
        print('select \'train\' or \'run\'')
        exit()

    # model options
    len_sequence = 14
    features_dim = 64
    hidden_dim = 32
    layer_dim = 1
    model = HighlightDetector(features_dim=features_dim,
                              hidden_dim=hidden_dim,
                              layer_dim=layer_dim,
                              resnet_pretrained=True)
    model_path = './highlight_detector_fd{}_hd{}_ld{}.ckpt'.format(str(features_dim),
                                                                   str(hidden_dim),
                                                                   str(layer_dim))

    option = sys.argv[1]
    if option == 'train':
        num_epochs = 100
        if len(sys.argv) > 2:
            num_epochs = int(sys.argv[2])
        batch_size = 32
        dataloaders = load_dataset(len_sequence=len_sequence, batch_size=batch_size, val_ratio=0.1)
        model = model.to(device)
        model_ft, hist = train_model(model,
                               dataloaders,
                               nn.MSELoss(),
                               torch.optim.Adam(model.parameters(), lr=0.001),
                               num_epochs=num_epochs)
        torch.save(model_ft.state_dict(), model_path)

        plt.xlabel("Training Epochs")
        plt.ylabel("Validation Accuracy")
        plt.plot(range(1, num_epochs + 1), hist)
        plt.ylim((0,1.))
        plt.xticks(np.arange(1, num_epochs + 1, 1.0))
        plt.legend()
        plt.show()
    elif option == 'run':
        if len(sys.argv) > 2:
            src_path = sys.argv[2]
        else:
            print('specify a video soruce path')
            exit()
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        video = cv2.VideoCapture(src_path)
        fwidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(num_frames / fps) * 1000
        interval_millis = 1000

        cur_frame = 0
        frames = []
        while(True):
            if cur_frame > num_frames:
                break
            if cur_frame * interval_millis > duration:
                break
            ret, frame = video.read()
            if ret:
                frame = cv2.resize(frame, dsize=(16 * 14, 9 * 14))
                if len(frames) == len_sequence:
                    frames.pop(0)
                    frames.append(frame)
                    transpoed_frames = np.asarray([[frame.transpose((2, 0, 1)) for frame in frames]])
                    inputs = torch.from_numpy(transpoed_frames).float()
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    score = outputs[0][0]
                    if score > 3.5:
                        print('time: {}m {}s, output: {}'.format(int(cur_frame / 60), int(cur_frame % 60), score))
                else:
                    frames.append(frame)
            cur_frame += 1
