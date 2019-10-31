
import cv2
import json
import os
import shutil
import re
from skimage.metrics import structural_similarity as ssim
from skimage import measure
from sklearn.cluster import KMeans

def frame_order(str):
    regex = re.compile('frames_([0-9]+)+_([0-9]+)+m_([0-9]+)s.jpg')
    try:
        match = regex.match(str)
        return int(match[1])
    except:
        return -1


def worst_similarity(imgseq):
    worst_sim = 1
    prev_img = None
    for img in imgseq:
        if prev_img is not None:
            sim = ssim(prev_img, img, multichannel=True)
            if worst_sim > sim:
                worst_sim = sim
        prev_img = img
    return worst_sim


def best_similarity(imgseq):
    best_sim = 0
    prev_img = None
    for img in imgseq:
        if prev_img is not None:
            sim = ssim(prev_img, img, multichannel=True)
            if best_sim < sim:
                best_sim = sim
        prev_img = img
    return best_sim


def exclude_too_similar_imgs(imgs_dir):
    regex = re.compile('frames_([0-9]+)+_([0-9]+)+m_([0-9]+)s.jpg')
    img_paths = []
    for filename in sorted(os.listdir(imgs_dir), key=frame_order):
        filepath = os.path.join(imgs_dir, filename)
        if os.path.isdir(filepath):
            continue
        if regex.match(filename) is not None:
            img_paths.append(filepath)

    ets_img_paths = []
    prev_img_path = None
    for cur_img_path in img_paths:
        exclude = False
        if prev_img_path is not None:
            cur_img, prev_img = cv2.imread(cur_img_path), cv2.imread(prev_img_path)
            if best_similarity([cur_img, prev_img]) > 0.85:
                exclude = True
            else:
                cur_match, prev_match = regex.match(os.path.basename(cur_img_path)), regex.match(os.path.basename(prev_img_path))
                cur_frame, cur_minutes, cur_seconds = int(cur_match[1]), int(cur_match[2]), int(cur_match[3])
                prev_frame, prev_minutes, prev_seconds = int(prev_match[1]), int(prev_match[2]), int(prev_match[3])
                if cur_frame - prev_frame < 5:
                    exclude = True
                if cur_minutes == prev_minutes and cur_seconds == prev_seconds:
                    exclude = cur_frame - prev_frame < 7
        if not exclude:
            ets_img_paths.append(cur_img_path)
            prev_img_path = cur_img_path
    return ets_img_paths


def delete_too_similar_imgs(imgs_dir):
    regex = re.compile('frames_([0-9]+)+_([0-9]+)+m_([0-9]+)s.jpg')
    img_paths = []
    for filename in sorted(os.listdir(imgs_dir), key=frame_order):
        filepath = os.path.join(imgs_dir, filename)
        if os.path.isdir(filepath):
            continue
        if regex.match(filename) is not None:
            img_paths.append(filepath)

    prev_img_path = None
    for cur_img_path in img_paths:
        exclude = False
        if prev_img_path is not None:
            cur_img, prev_img = cv2.imread(cur_img_path), cv2.imread(prev_img_path)
            if best_similarity([cur_img, prev_img]) > 0.85:
                exclude = True
            else:
                cur_match, prev_match = regex.match(os.path.basename(cur_img_path)), regex.match(os.path.basename(prev_img_path))
                cur_frame, cur_minutes, cur_seconds = int(cur_match[1]), int(cur_match[2]), int(cur_match[3])
                prev_frame, prev_minutes, prev_seconds = int(prev_match[1]), int(prev_match[2]), int(prev_match[3])
                if cur_frame - prev_frame < 5:
                    exclude = True
                if cur_minutes == prev_minutes and cur_seconds == prev_seconds:
                    exclude = cur_frame - prev_frame < 3
        if not exclude:
            prev_img_path = cur_img_path
        else:
            print('{} removed'.format(cur_img_path))
            os.remove(cur_img_path)


def kmeans_grouping(imgs_dir):
    regex = re.compile('frames_([0-9]+)+_([0-9]+)+m_([0-9]+)s.jpg')
    X = []
    X_map = {}
    img_paths = exclude_too_similar_imgs(imgs_dir)
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        match = regex.match(img_name)
        try:
            frame, minutes, seconds = int(match[1]), int(match[2]), int(match[3])
            seconds = 60 * minutes + seconds
            X.append([frame, seconds])
            X_map[frame] = img_path
        except:
            continue
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, max_iter=20).fit(X)

    series_dir = os.path.join(imgs_dir, 'series')
    if not os.path.exists(series_dir):
        os.makedirs(series_dir)
    for i in range(n_clusters):
        os.makedirs(os.path.join(series_dir, str(i)))
    for idx in range(len(kmeans.labels_)):
        frame = X[idx][0]
        imgpath = X_map[frame]
        label = kmeans.labels_[idx]
        trg_path = os.path.join(series_dir, str(label), os.path.basename(imgpath))
        shutil.copy(imgpath, trg_path)


# imgs in series directory
def concat_imgs(imgs_dir, trg_dir):
    regex = re.compile('frames_([0-9]+)+_([0-9]+)+m_([0-9]+)s.jpg')
    len_sequence = 7
    img_paths = []
    for filename in sorted(os.listdir(imgs_dir), key=frame_order):
        filepath = os.path.join(imgs_dir, filename)
        if os.path.isdir(filepath):
            continue
        if regex.match(filename) is not None:
            img_paths.append(filepath)

    for i in range(0, len(img_paths) - len_sequence):
        dirname = os.path.basename(os.path.dirname(img_paths[i].split('series/')[0]))
        img_seq_paths = [img_paths[i + order] for order in range(len_sequence)]
        imgs = [cv2.imread(img_path) for img_path in img_seq_paths]
        worst_sim = worst_similarity(imgs)
        if worst_sim < 0.3:
            print('{} failed. worst similarity: {}'.format(img_seq_paths[0], worst_sim))
            continue
        match = regex.match(os.path.basename(img_paths[i]))
        try:
            frame = int(match[1])
            trg_path = os.path.join(trg_dir, '{}_{}.jpg'.format(dirname, frame))
            cv2.imwrite(trg_path, cv2.hconcat(imgs))
        except:
            continue


def delete_highlight_imgs_in_fulltime(imgs_dir, highlight_scenes):
    regex = re.compile('frames_([0-9]+)+_([0-9]+)+m_([0-9]+)s.jpg')
    for filename in sorted(os.listdir(imgs_dir), key=frame_order):
        filepath = os.path.join(imgs_dir, filename)
        match = regex.match(filename)
        try:
            frame, minutes, seconds = int(match[1]), int(match[2]), int(match[3])
            seconds = 60 * minutes + seconds
            is_highlight = False
            for highlight_scene in highlight_scenes:
                if highlight_scene['start_time'] < seconds < highlight_scene['end_time']:
                    is_highlight = True
                    break
            if is_highlight:
                print('{} removed'.format(filepath))
                os.remove(filepath)
        except:
            continue


def delete_highlight_imgs_in_fulltimes(game_dir, highlight_dir):
    for game_name in os.listdir(game_dir):
        game_path = os.path.join(game_dir, game_name)
        json_path = os.path.join(highlight_dir, game_name + '.json')
        if not os.path.exists(json_path):
            print('{} does not exist.'.format(json_path))
            continue
        with open(json_path) as fjson:
            djson = json.load(fjson)
            delete_highlight_imgs_in_fulltime(game_path, djson['highlight_scenes'])


def extract_highlight_imgs_with_group(highlight_imgs_dir):
    for game_name in os.listdir(highlight_imgs_dir):
        game_path = os.path.join(highlight_imgs_dir, game_name)
        print('remove unnecessary directories')
        for filename in sorted(os.listdir(game_path)):
            filepath = os.path.join(game_path, filename)
            if os.path.isdir(filepath):
                shutil.rmtree(filepath)
        print('kmeans: {}'.format(game_path))
        kmeans_grouping(game_path)

        series_dir_path = os.path.join(game_path, 'series')
        concat_trgdir_path = os.path.join(game_path, 'result')
        if os.path.exists(concat_trgdir_path):
            shutil.rmtree(concat_trgdir_path)
        os.makedirs(concat_trgdir_path)
        print('start concating images: {}'.format(series_dir_path))
        for series_name in os.listdir(series_dir_path):
            series_path = os.path.join(series_dir_path, series_name)
            concat_imgs(series_path, concat_trgdir_path)


def concat_highlight_imgs(highlight_imgs_dir):
    for game_name in os.listdir(highlight_imgs_dir):
        game_path = os.path.join(highlight_imgs_dir, game_name)
        series_dir_path = os.path.join(game_path, 'series')
        concat_trgdir_path = os.path.join(game_path, 'result')
        if os.path.exists(concat_trgdir_path):
            shutil.rmtree(concat_trgdir_path)
        os.makedirs(concat_trgdir_path)
        print('start concating images: {}'.format(series_dir_path))
        for series_name in os.listdir(series_dir_path):
            series_path = os.path.join(series_dir_path, series_name)
            concat_imgs(series_path, concat_trgdir_path)


def concat_non_highlight_imgs(non_highlight_dir):
    for game_name in os.listdir(non_highlight_dir):
        game_path = os.path.join(non_highlight_dir, game_name)
        # delete_too_similar_imgs(game_path)
        concat_trgdir_path = os.path.join(game_path, 'result')
        if os.path.exists(concat_trgdir_path):
            shutil.rmtree(concat_trgdir_path)
        os.makedirs(concat_trgdir_path)
        print('start concating images: {}, {}'.format(game_path, concat_trgdir_path))
        concat_imgs(game_path, concat_trgdir_path)


def delete_too_similar_img_in_one(imgs_dir):
    len_sequence = 7
    for imgname in sorted(os.listdir(imgs_dir)):
        imgpath = os.path.join(imgs_dir, imgname)
        img = cv2.imread(imgpath)
        imgs = [img[:, i * 16 * 14 : (i + 1) * 16 * 14, :] for i in range(len_sequence)]
        worst_sim = worst_similarity(imgs)
        if worst_sim < 0.4:
            os.remove(imgpath)


if __name__ == "__main__":
    full_dir = './raw_files/full/'
    full_imgs_dir = os.path.join(full_dir, 'images')
    highlight_dir = './raw_files/highlight/'
    highlight_imgs_dir = os.path.join(highlight_dir, 'images')
    non_highlight_dir = './raw_files/non-highlight'

    # for dirname, subdirlist, filelist in os.walk(highlight_imgs_dir):
    #     if os.path.basename(dirname) == 'result':
    #         for fname in filelist:
    #             img_path = os.path.join(dirname, fname)
    #             trgdir = os.path.join(highlight_dir, 'result')
    #             trgpath = os.path.join(trgdir, fname)
    #             if not os.path.exists(trgdir):
    #                 os.makedirs(trgdir)
    #             shutil.copy(img_path, trgpath)
    
    # for dirname, subdirlist, filelist in os.walk(non_highlight_dir):
    #     if os.path.basename(dirname) == 'result':
    #         for fname in filelist:
    #             img_path = os.path.join(dirname, fname)
    #             trgdir = os.path.join(non_highlight_dir, 'result')
    #             trgpath = os.path.join(trgdir, fname)
    #             if not os.path.exists(trgdir):
    #                 os.makedirs(trgdir)
    #             shutil.copy(img_path, trgpath)

    # extract_highlight_imgs_with_group(highlight_imgs_dir)
    # concat_highlight_imgs(highlight_imgs_dir)
    
    # delete_highlight_imgs_in_fulltimes(non_highlight_dir, highlight_dir)
    # concat_non_highlight_imgs(non_highlight_dir)
