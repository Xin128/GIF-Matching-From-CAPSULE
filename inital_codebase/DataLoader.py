"""
Initial loading: 100 gifs
3. The general query pipeline we have for gif query is following:
    1. Input: a query gif (may contain multiple frames)
    2. Extract the gif to multiple images based on frame number (currently using PIL IMAGE package)
    3. For each image frame in gif, query the top similar image from training gif-image collections using CAPSULE.
    4. Rank potential gifs based on all similar images’ corresponding gifID frequencies.
    5. Return the top k similar gif id.
4. Training pipeline:
    1. For each gif in collections, split it into multiple frames.
    Apply feature extraction on each image frame (same as CAPSULE process).
    Insert gif id in our hash table.

"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import requests
import cv2

class DataLoader():
    def __init__(self):
        self.numImages = 0
        self.imageUrls = []

    def readImages(self):
        """
        update imageUrls from tsv files
        :return:
        """
        gif_file = open('raw_gifs.tsv')
        lines = gif_file.readlines()
        self.numImages = len(lines)
        for line in lines:
            self.imageUrls.append(line.split()[0])

        for i in range(10):
            print(i)
            response = requests.get(self.imageUrls[i])
            # response = requests.get('https://38.media.tumblr.com/17bdef6f42954defdd0b250a5788e54a/tumblr_nf21kqOJiD1siwn55o1_500.gif')
            image_bytes = io.BytesIO(response.content)
            im = Image.open(image_bytes)
            if self.removeDuplicates(im) == -1:
                continue
            print('------------------------------------------------------------------------')

    def removeDuplicates(self, im):
        """

        1) remove duplicate image for the frame
        :return: [[]]
        """
        array_lst = []
        image_lst = []
        # To iterate through the entire gif
        numframes = 0
        try:
            while 1:
                im.seek(im.tell() + 1)
                numframes += 1
        except EOFError:
            im.seek(0)
        interval = numframes // 10 + 1
        print("numframes", numframes, interval)
        try:
            while 1:
                if (im.tell() % interval == 0):
                    new = Image.new("RGBA", im.size)
                    new.paste(im)
                    arr = np.array(new).astype(np.uint8)
                    array_lst.append(arr)
                    image_lst.append(np.array(new))
                im.seek(im.tell() + 1)
        except EOFError:
            pass
        if len(image_lst) > 10:
            return -1

        # show sampled image here
        # for i in array_lst:
        #     Image.fromarray(i).show()

dl = DataLoader()
dl.readImages()
