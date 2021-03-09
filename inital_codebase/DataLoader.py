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
import requests

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

        for i in range(self.numImages):
            im = Image.open(requests.get(self.imageUrls[i]))
            print(im)
            print('------------------------------------------------------------------------')

    def removeDuplicates(self):
        """

        1) remove duplicate image for the frame
        :return: [[]]
        """


dl = DataLoader()
dl.readImages()
