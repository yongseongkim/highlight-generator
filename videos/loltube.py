# -*- coding: utf8 -*-
import os
import json
from pytube import YouTube
from google.cloud import storage
from google.cloud.storage import Blob

curdirpath = os.path.dirname(os.path.realpath(__file__))
client = storage.Client.from_service_account_json(os.path.join(curdirpath, 'gcp-key.json'))
bucket = client.get_bucket("lol-videos")

with open(os.path.join(curdirpath, 'videos.json')) as data:
    videos = json.load(data)
    for key in videos.keys():
        print('request: {}'.format(key))
        if os.path.exists('{}.mp4'.format(key)):
            print('{} already exsits'.format(key))
            continue
        outputdirpath = os.path.join(curdirpath, 'raw_files')
        YouTube(videos[key]).streams.first().download(output_path=outputdirpath, filename=key)
        filepath = os.path.join(outputdirpath, '{}.mp4'.format(key))
        if os.path.exists(filepath):
            print('upload to gcp: {}'.format(key))
            Blob(os.path.basename(filepath), bucket).upload_from_filename(filepath)
