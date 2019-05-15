# -*- coding: utf8 -*-
import os
import json
import sys
import time
from pytube import YouTube
from google.cloud import storage
from google.cloud.storage import Blob

curdirpath = os.path.dirname(os.path.realpath(__file__))
client = storage.Client.from_service_account_json(os.path.join(curdirpath, 'gcp-key.json'))
bucket = client.get_bucket("lol-videos")

with open(os.path.join(curdirpath, 'full.json')) as data:
	videos = json.load(data)
	for key in videos.keys():
		if os.path.exists('{}.mp4'.format(key)):
			print('{} already exsits in directory'.format(key))
			continue
		outputdirpath = os.path.join(curdirpath, 'raw_files')
		if not os.path.exists(outputdirpath):
			os.makedirs(outputdirpath)
		filepath = os.path.join(outputdirpath, '{}.mp4'.format(key))
		blob = Blob(os.path.join('full', os.path.basename(filepath)), bucket)
		try:
			if not blob.exists():
				print('try to download: {}'.format(key))
				YouTube(videos[key]).streams.first().download(output_path=outputdirpath, filename=key)
				print('try to upload to gcp: {}'.format(key))
				blob.upload_from_filename(filepath)
				os.remove(filepath)	
				print('removed file: {}'.format(filepath))
		except:
			e = sys.exc_info()[0]
			print(key, e)

