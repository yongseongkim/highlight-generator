import cv2
import os
import numpy as np
import time
import sys
from PIL import Image


def frame_order(str):
    try:
        return int(os.path.splitext(str)[0].split('_')[1])
    except:
        return -1


def preprocess(img):
    # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return grayscale
    return img


def extract_frames(src_path, dest_dir, size=None, interval_millis=1000, st_margin_miilis=0, et_margin_millis=0):
    if not os.path.exists(src_path) or not os.path.isfile(src_path):
        print('There is no source file.')
        raise ValueError

    print('###### start extracting frames: %s' % (src_path))
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
    while(True):
        if cur_frame > num_frames:
            break
        if interval_millis is not None:
            start_time = (st_margin_miilis or 0) + \
                (cur_frame * interval_millis)
            if (start_time > (duration - et_margin_millis)) or start_time > duration:
                break
            if cur_frame > num_frames / ((fps * interval_millis) / 1000):
                break
            video.set(cv2.CAP_PROP_POS_MSEC, start_time)
            oname = '{}_{}'.format(
                filename, 'frame_interval_millis_' + str(start_time))
        else:
            oname = '{}_{}'.format(filename, 'frame_' + str(cur_frame))
        oname += '.jpg'
        ret, frame = video.read()
        if ret:
            frame = preprocess(frame)
            if size is None:
                size = (fwidth, fheight)
            resized = cv2.resize(frame, dsize=size)
            output_dir = os.path.join(dest_dir, filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(os.path.join(output_dir, oname), resized)
        else:
            break
        cur_frame += 1
    video.release()
    cv2.destroyAllWindows()
    print('###### complete extracting frames from %s' % src_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('source path, output destination directory must not empty.')
        exit()
    src_path = sys.argv[1]
    dest_dir = sys.argv[2]
    extract_frames(src_path=src_path,
                   dest_dir=dest_dir,
                   size=(16 * 14, 9 * 14),
                   interval_millis=1000,
                   st_margin_miilis=1260000,
                   et_margin_millis=2890000)
