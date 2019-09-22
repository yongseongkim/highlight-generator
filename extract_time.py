#
# 게임 영상에서 게임 진행 시간 추출하기
#
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
        # 롤 최장 경기 기록은 94분 40초
        self._duration = 95 * 60
        self._times = [0 for x in range(self._duration)]
        # 같은 구간으로 판단하는 단위, 5초 이내는 같은 구간으로 처리한다.
        self._threshold = 5

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

    def to_dict(self):
        highlight_scenes = []
        sect_st = -1
        for idx in range(self._duration):
            if sect_st == -1 and self._times[idx] == 1:
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
    return img


def extract_time(src_path, interval_millis=1000, start_millis=None, end_millis=None):
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
    duration = int(num_frames / fps)
    print('###### %d x %d, fps: %f, the number of frames: %d, duration: %d seconds' % (fwidth, fheight, fps, num_frames, duration))

    cur_frame = 0
    video.set(cv2.CAP_PROP_POS_MSEC, start_millis)
    hl = Higlight(name=filename)
    while(True):
        if interval_millis is not None:
            start_time = (start_millis or 0) + (cur_frame * interval_millis)
            video.set(cv2.CAP_PROP_POS_MSEC, start_time)
            if start_time > (end_millis or ((num_frames / fps) * 1000)):
                break
            if cur_frame > num_frames / ((fps * interval_millis) / 1000):
                break
        if cur_frame > num_frames:
            break
        ret, frame = video.read()
        if ret:
            frame = preprocess(frame)
            cwidth, cheight = 11, 18

            # front (78, 942, 20, 24) / back (78, 968, 20, 23)
            img_dir = '/Users/yongseongkim/Documents/workspace.nosync/highlight-generator/highlight_frames/'
            cx, cy = 945, 79
            ff_img = frame[cy: cy + cheight, cx: cx + cwidth]
            cv2.imwrite(os.path.join(img_dir, 'ff_frame_' + str(cur_frame) + '.jpg'), ff_img)
            ff_text = pytesseract.image_to_string(
                Image.fromarray(ff_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            cx, cy = 954, 79
            fb_img = frame[cy: cy + cheight, cx: cx + cwidth]
            cv2.imwrite(os.path.join(img_dir, 'fb_frame_' + str(cur_frame) + '.jpg'), fb_img)
            fb_text = pytesseract.image_to_string(
                Image.fromarray(fb_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')

            cx, cy = 968, 79
            bf_img = frame[cy: cy + cheight, cx: cx + cwidth]
            cv2.imwrite(os.path.join(img_dir, 'bf_frame_' + str(cur_frame) + '.jpg'), bf_img)
            bf_text = pytesseract.image_to_string(
                Image.fromarray(bf_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            cx, cy = 977, 79
            bb_img = frame[cy: cy + cheight, cx: cx + cwidth]
            cv2.imwrite(os.path.join(img_dir, 'bb_frame_' + str(cur_frame) + '.jpg'), bb_img)
            bb_text = pytesseract.image_to_string(
                Image.fromarray(bb_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            
            print('current frame: %d, front text : %s, %s, back text: %s, %s' % (cur_frame, ff_text, fb_text, bf_text, bb_text))
            try:
                hl.add_time(ff_text, fb_text, bf_text, bb_text)
            except Exception as e:
                print('Error happend from %s, the order of frame is %d, interval millis is %d, start millis is %d, end millis is %d\n'
                      % (src_path, cur_frame, interval_millis, start_millis, end_millis), e)
        else:
            break
        cur_frame += 1
    video.release()
    cv2.destroyAllWindows()

    with open(os.path.join(os.path.dirname(src_path), '%s.json' % filename), 'w') as fp:
        json.dump(hl.to_dict(), fp)
    print('###### complete extracting time from %s' % src_path)


extract_time(src_path='/Users/yongseongkim/Downloads/highlight.webm',
             interval_millis=None,
             start_millis=25000,
             end_millis=100000)
