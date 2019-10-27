# -*- coding: utf8 -*-
import os
import json
import sys
import time
from pytube import YouTube

curdirpath = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(curdirpath, 'full-summer.json')) as data:
    videos = json.load(data)
    for key in videos.keys():
        outputdirpath = os.path.join(curdirpath, 'raw_files', 'full')
        if not os.path.exists(outputdirpath):
            os.makedirs(outputdirpath)
        filepath = os.path.join(outputdirpath, '{}.webm'.format(key))
        if os.path.exists(filepath):
            continue
        print('try to download: {}'.format(key))
        YouTube(videos[key]).streams\
            .filter(adaptive=True, only_video=True)\
            .order_by('resolution').desc().first()\
            .download(output_path=outputdirpath, filename=key)
