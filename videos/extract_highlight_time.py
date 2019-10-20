#
# 게임 영상에서 게임 진행 시간 추출하기
#
from pytube import YouTube
import cv2
import os
import numpy as np
import uuid
import time
import pytesseract
import re
import json
from PIL import Image


class Higlight:
    def __init__(self, name):
        self._name = name
        self._duration = 100 * 60
        self._times = [0 for x in range(self._duration)]
        self._fill_window = 10
        self._remove_threshold = 5

    def add_time(self, ff, fb, bf, bb):
        if not ff or not fb or not bf or not bb:
            return
        iff, ifb, ibf, ibb = int(ff), int(fb), int(bf), int(bb)
        minutes, seconds = (10 * iff + ifb), (10 * ibf + ibb)
        if seconds > 60:
            return
        seconds = (60 * minutes) + seconds
        if seconds > self._duration:
            return
        self._times[seconds] += 1

    def fill_if_has_adjacent(self):
        for idx in range(self._duration - self._fill_window):
            lidx = -1
            ridx = -1
            for w in range(self._fill_window):
                if self._times[idx + w] >= 1:
                    lidx = idx + w
                    break
            for w in range(self._fill_window):
                if self._times[idx + self._fill_window - w] >= 1:
                    ridx = idx + self._fill_window - w
                    break
            if lidx != -1 and ridx != -1:
                for w in range(lidx, ridx, 1):
                    self._times[w] = 1

    def remove_if_not_has_adjacent(self):
        idx = 0
        s0 = []
        s1 = []
        for idx in range(self._duration):
            if self._times[idx] == 0:
                if len(s1) >= self._remove_threshold:
                    s1 = []
                s0.append(idx)
            else:
                if len(s0) > self._remove_threshold and 0 < len(s1) < self._remove_threshold:
                    for i in s1:
                        self._times[i] = 0
                    s1 = []
                s0 = []
                s1.append(idx)
        if 0 < len(s1) < self._remove_threshold:
            for i in s1:
                self._times[i] = 0

    def to_dict(self):
        self.remove_if_not_has_adjacent()
        self.fill_if_has_adjacent()

        highlight_scenes = []
        sect_st = -1
        for idx in range(self._duration):
            if sect_st == -1 and self._times[idx] >= 1:
                sect_st = idx
            elif sect_st >= 0 and self._times[idx] == 0:
                highlight_scenes.append({
                    'start_time': sect_st,
                    'end_time': idx-1
                })
                sect_st = -1
        return {
            'name': self._name,
            'highlight_scenes': highlight_scenes
        }


def frame_order(str):
    try:
        return int(os.path.splitext(str)[0].split('_')[1])
    except:
        return -1


def preprocess(img):
    # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return grayscale
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,21)
    return denoised
    # return img


def extract_time(src_path, interval_millis=1000, st_margin_miilis=None, et_margin_millis=None):
    if not os.path.exists(src_path) or not os.path.isfile(src_path):
        print('there is no source.')
        raise ValueError

    print('###### start extracting time: %s' % (src_path))
    filename = os.path.splitext(os.path.basename(src_path))[0]
    video = cv2.VideoCapture(src_path)
    fwidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = int(num_frames / fps) * 1000
    print('###### %d x %d, fps: %f, the number of frames: %d, duration: %d seconds' % (
        fwidth, fheight, fps, num_frames, duration))

    cur_frame = 0
    video.set(cv2.CAP_PROP_POS_MSEC, st_margin_miilis)
    hl = Higlight(name=filename)
    while(True):
        if interval_millis is not None:
            start_time = (st_margin_miilis or 0) + \
                (cur_frame * interval_millis)
            video.set(cv2.CAP_PROP_POS_MSEC, start_time)
            if start_time > ((duration - et_margin_millis) or ((num_frames / fps) * 1000)):
                break
            if cur_frame > num_frames / ((fps * interval_millis) / 1000):
                break
        if cur_frame > num_frames:
            break
        ret, frame = video.read()
        if ret:
            frame = preprocess(frame)
            cwidth, cheight = 10, 18

            # spring: (944, 78) (953, 78) (967, 78) (976, 78)
            # summer: (945, 78) (954, 78) (968, 78) (977, 89)
            # img_dir = './highlight_frames/'
            # if not os.path.exists(img_dir):
            #     os.makedirs(img_dir)
            cx, cy = 945, 78
            ff_img = frame[cy: cy + cheight, cx: cx + cwidth]
            # cv2.imwrite(os.path.join(img_dir, 'ff_frame_' + str(cur_frame) + '.jpg'), ff_img)
            ff_text = pytesseract.image_to_string(
                Image.fromarray(ff_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            cx, cy = 954, 78
            fb_img = frame[cy: cy + cheight, cx: cx + cwidth]
            # cv2.imwrite(os.path.join(img_dir, 'fb_frame_' + str(cur_frame) + '.jpg'), fb_img)
            fb_text = pytesseract.image_to_string(
                Image.fromarray(fb_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')

            cx, cy = 968, 78
            bf_img = frame[cy: cy + cheight, cx: cx + cwidth]
            # cv2.imwrite(os.path.join(img_dir, 'bf_frame_' + str(cur_frame) + '.jpg'), bf_img)
            bf_text = pytesseract.image_to_string(
                Image.fromarray(bf_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            cx, cy = 977, 78
            bb_img = frame[cy: cy + cheight, cx: cx + cwidth]
            # cv2.imwrite(os.path.join(img_dir, 'bb_frame_' + str(cur_frame) + '.jpg'), bb_img)
            bb_text = pytesseract.image_to_string(
                Image.fromarray(bb_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            print('current frame: %d, front text : %s, %s, back text: %s, %s' %
                  (cur_frame, ff_text, fb_text, bf_text, bb_text))
            try:
                hl.add_time(ff_text, fb_text, bf_text, bb_text)
            except Exception as e:
                print('Error happend from %s, the order of frame is %d, interval millis is %d, start millis is %d, end millis is %d\n'
                      % (src_path, cur_frame, interval_millis, st_margin_miilis, duration - et_margin_millis), e)
        else:
            break
        cur_frame += 1
    video.release()
    cv2.destroyAllWindows()

    with open(os.path.join(os.path.dirname(src_path), '%s.json' % filename), 'w') as fp:
        json.dump(hl.to_dict(), fp)
    print('###### complete extracting time from %s' % src_path)


if __name__ == "__main__":    
    video_dir = './raw_files/'
    with open('./highlight-summer.json') as data:
        videos = json.load(data)
        for key in videos:
            filepath = os.path.join(video_dir, key + '.webm')
            if os.path.exists(filepath):
                continue
            print('##### download file: %s, %s' % (key, videos[key]))
            try:
                YouTube(videos[key]).streams.filter(adaptive=True, only_video=True).order_by(
                    'resolution').desc().first().download(output_path=video_dir, filename=key)
                if os.path.exists(filepath):
                    extract_time(src_path=filepath, interval_millis=None,
                                st_margin_miilis=15000, et_margin_millis=15000)
            except Exception as e:
                print(e)
