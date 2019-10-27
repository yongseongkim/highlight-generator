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


def preprocess(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayscale_clahe = clahe.apply(grayscale)
    return grayscale_clahe
    # grayscale_equ = cv2.equalizeHist(grayscale)
    # return grayscale_equ



def extract_frames(src_path, output_dir, interval_millis=1000, st_margin_miilis=None, et_margin_millis=None):
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
            processed_img = preprocess(frame)
            cwidth, cheight = 10, 18

            # spring: (944, 78) (953, 78) (967, 78) (976, 78)
            # summer: (945, 78) (954, 78) (968, 78) (977, 89)
            cx, cy = 945, 78
            ff_img = processed_img[cy: cy + cheight, cx: cx + cwidth]
            ff_text = pytesseract.image_to_string(
                Image.fromarray(ff_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            cx, cy = 954, 78
            fb_img = processed_img[cy: cy + cheight, cx: cx + cwidth]
            fb_text = pytesseract.image_to_string(
                Image.fromarray(fb_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            cx, cy = 968, 78
            bf_img = processed_img[cy: cy + cheight, cx: cx + cwidth]
            bf_text = pytesseract.image_to_string(
                Image.fromarray(bf_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            cx, cy = 977, 78
            bb_img = processed_img[cy: cy + cheight, cx: cx + cwidth]
            # cv2.imwrite(os.path.join(img_dir, 'bb_frame_' + str(cur_frame) + '.jpg'), bb_img)
            bb_text = pytesseract.image_to_string(
                Image.fromarray(bb_img),
                lang='eng',
                config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
            print('current frame: %d, front text : %s, %s, back text: %s, %s' % (cur_frame, ff_text, fb_text, bf_text, bb_text))
            try:
                if not ff_text or not fb_text or not bf_text or not bb_text:
                    cur_frame += 1
                    continue
                iff, ifb, ibf, ibb = int(ff_text), int(fb_text), int(bf_text), int(bb_text)
                minutes, seconds = (10 * iff + ifb), (10 * ibf + ibb)
                if seconds > 60:
                    raise ValueError
                if (60 * minutes) + seconds > 90 * 60:
                    raise ValueError
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, 'frames_{}_{}m_{}s.jpg'.format(str(cur_frame), str(minutes), str(seconds)))
                resized = cv2.resize(frame, dsize=(16 * 14, 9 * 14), interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_path, resized)
            except Exception as e:
                print('Error happend from %s, the order of frame is %d, interval millis is %d, start millis is %d, end millis is %d\n' % (src_path, cur_frame, interval_millis, st_margin_miilis, duration - et_margin_millis), e)
        else:
            break
        cur_frame += 1
    video.release()
    cv2.destroyAllWindows()
    print('###### complete extracting time from %s' % src_path)


if __name__ == "__main__":
    dirname = './raw_files/'
    for filename in os.listdir(dirname):
        filepath = os.path.join(dirname, filename)
        if os.path.isdir(filepath):
            continue
        outname = os.path.splitext(os.path.basename(filename))[0]
        outdir = os.path.join('./raw_files/images/', outname)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            continue
        if os.path.splitext(filepath)[1] == '.webm':
            extract_frames(filepath, outdir, interval_millis=100,
                           st_margin_miilis=15000, et_margin_millis=15000)
