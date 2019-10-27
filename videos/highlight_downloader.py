# -*- coding: utf8 -*-
import os
import json
import time
from pytube import YouTube

curdirpath = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(curdirpath, 'highlight-summer.json')) as data:
    videos = json.load(data)
    for key in videos.keys():
        outputdirpath = os.path.join(curdirpath, 'raw_files', 'highlight')
        if not os.path.exists(outputdirpath):
            os.makedirs(outputdirpath)
        filepath = os.path.join(outputdirpath, '{}.webm'.format(key))
        if os.path.exists(filepath):
            continue
        print('try to download: {}, {}'.format(key, videos[key]))
        YouTube(videos[key]).streams\
                            .filter(adaptive=True, only_video=True)\
                            .order_by('resolution').desc().first()\
                            .download(output_path=outputdirpath, filename=key)
