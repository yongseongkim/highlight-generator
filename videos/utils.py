# similarity

import cv2
import os
from skimage import measure

img_dir = './raw_files/highlight/images/lckspringsplit_af_dwg_week_6_game_1'
img1 = cv2.imread(os.path.join(img_dir, 'frames_191_5m_55s.jpg'))
img2 = cv2.imread(os.path.join(img_dir, 'frames_192_5m_55s.jpg'))
img3 = cv2.imread(os.path.join(img_dir, 'frames_193_5m_55s.jpg'))
img4 = cv2.imread(os.path.join(img_dir, 'frames_228_1m_55s.jpg'))
img5 = cv2.imread(os.path.join(img_dir, 'frames_229_7m_46s.jpg'))

diff12 = measure.compare_ssim(img1, img2, multichannel=True)
diff23 = measure.compare_ssim(img2, img3, multichannel=True)
diff13 = measure.compare_ssim(img1, img3, multichannel=True)
diff34 = measure.compare_ssim(img3, img4, multichannel=True)
diff45 = measure.compare_ssim(img4, img5, multichannel=True)

# grouping

import os
import shutil
import re
from sklearn.cluster import KMeans

root_dir = './raw_files/highlight/images'
regex = re.compile('frames_([0-9]+)+_([0-9]+)+m_([0-9]+)s.jpg')

for gamedir_name in os.listdir(root_dir):
    gamedir_path = os.path.join(root_dir, gamedir_name)
    imgnames = sorted(os.listdir(gamedir_path))
    if len(imgnames) == 0:
        shutil.rmtree(gamedir_path)
        continue
    
    X = []
    X_map = {}
    for imgname in imgnames:
        imgpath = os.path.join(gamedir_path, imgname)
        match = regex.match(imgname)
        try:
            frame, minutes, seconds = int(match[1]), int(match[2]), int(match[3])
            seconds = 60 * minutes + seconds
            X.append([frame, seconds])
            X_map[frame] = imgpath
        except:
            continue
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, max_iter=20).fit(X)
    for i in range(n_clusters):
        os.makedirs(os.path.join(gamedir_path, 'series_{}'.format(str(i))))
    for idx in range(len(kmeans.labels_)):
        frame = X[idx][0]
        imgpath = X_map[frame]
        label = kmeans.labels_[idx]
        trg_path = os.path.join(gamedir_path, 'series_{}'.format(label), os.path.basename(imgpath))
        shutil.copy(imgpath, trg_path)

# remove too similar image in series

import os
import cv2
from skimage import measure

root_dir = './raw_files/highlight/images'
for gamedir_name in os.listdir(root_dir):
    gamedir_path = os.path.join(root_dir, gamedir_name)
    filenames = sorted(os.listdir(gamedir_path))
    for filename in filenames:
        filepath = os.path.join(gamedir_path, filename)
        if not os.path.isdir(filepath) or not 'series' in filename:
            continue
        imgs = sorted(os.listdir(filepath))
        prev_img = None
        for imgname in imgs:
            cur_img_path = os.path.join(filepath, imgname)
            cur_img = cv2.imread(cur_img_path)
            if prev_img is not None:
                sim = measure.compare_ssim(prev_img, cur_img, multichannel=True)
                if sim > 0.90:
                    print(cur_img_path)
                    # os.remove(cur_img_path)
                    continue
            prev_img = cur_img

# concat image

import os
import cv2
import re
from skimage import measure

regex = re.compile('frames_([0-9]+)+_([0-9]+)+m_([0-9]+)s.jpg')
len_sequence = 7
root_dir = './raw_files/highlight/images'
for gamedir_name in os.listdir(root_dir):
    gamedir_path = os.path.join(root_dir, gamedir_name)
    filenames = sorted(os.listdir(gamedir_path))
    for filename in filenames:
        series_path = os.path.join(gamedir_path, filename)
        if not os.path.isdir(series_path) or not 'series' in series_path:
            continue
        imgnames = sorted(os.listdir(series_path))
        for i in range(0, len(imgnames) - len_sequence):
            img_paths = [os.path.join(series_path, imgnames[order + i]) for order in range(len_sequence)]
            imgs = [cv2.imread(img_path) for img_path in img_paths]
            prev_img = None
            worst_sim = 1
            for img in imgs:
                if prev_img is not None:
                    sim = measure.compare_ssim(prev_img, img, multichannel=True)
                    if worst_sim > sim:
                        worst_sim = sim
                prev_img = img
            if worst_sim < 0.4:
                continue
            match = regex.match(os.path.basename(img_paths[0]))
            try:
                frame = int(match[1])
                filepath = gamedir_path + '_highlight_' + str(frame) + '.jpg'
                concat_img = cv2.hconcat(imgs)
                cv2.imwrite(filepath, concat_img)
            except:
                continue
