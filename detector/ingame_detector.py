from models import IngameDetectorModel
import torch
import numpy as np
import cv2
import os
import sys
from dataclasses import dataclass
import torchvision.transforms as transforms


@dataclass
class IngameSection:
    start_millis: int
    end_millis: int


class IngameDetector:
    model = IngameDetectorModel()

    def __init__(self, model_path=None):
        # load model
        device = torch.device('cpu')
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))

    def detect(self, video_path, interval_millis=1000):
        dtransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        video = cv2.VideoCapture(video_path)
        # fwidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(num_frames / fps) * 1000

        cur_frame = 600
        while(True):
            start_time = cur_frame * interval_millis
            if cur_frame > num_frames:
                break
            if cur_frame * interval_millis > duration:
                break
            video.set(cv2.CAP_PROP_POS_MSEC, start_time)
            ret, frame = video.read()
            if ret:
                resized = cv2.resize(frame, dsize=(16 * 16, 9 * 16))
                timg = dtransforms(resized)
                inputs = timg.clone().detach().unsqueeze(0).float()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                result = predicted.tolist()[0]
                print('time: {}, ingame: {}'.format(start_time, result))
            cur_frame += 1
        video.release()
        cv2.destroyAllWindows()
        return []


if __name__ == "__main__":
    detector = IngameDetector(model_path='./models/ingame_detector.ckpt')
    if len(sys.argv) < 2:
        print('source path, output destination directory must not empty.')
        exit()
    src_path = sys.argv[1]
    detected = detector.detect(src_path)
    print(detected)
