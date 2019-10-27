import cv2
import os
from skimage import measure

img_dir = './raw_files/images/lckspringsplit_af_dwg_week_6_game_1'
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

root_dir = './raw_files/images'
for imgdir_name in os.listdir(root_dir):
    imgdir_path = os.path.join(root_dir, imgdir_name)
    for imgname in sorted(os.listdir(imgdir_path)):
        imgpath = os.path.join(imgdir_path, imgname)
        print(imgpath)
