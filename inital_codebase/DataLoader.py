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
import random
import cv2


alg = cv2.AKAZE_create()



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

        for i in range(1):
            print("CHEK!!!!")
            print(i,self.imageUrls[i] )
            response = requests.get(self.imageUrls[i])
            # response = requests.get('https://38.media.tumblr.com/17bdef6f42954defdd0b250a5788e54a/tumblr_nf21kqOJiD1siwn55o1_500.gif')
            image_bytes = io.BytesIO(response.content)
            im = Image.open(image_bytes)
            img_data_lst = self.removeDuplicates(im)
            if img_data_lst == None:
                continue
            else:
                for img_data in img_data_lst:
                    # Dinding image keypoints
                    kps = alg.detect(img_data)
                    # Getting first 32 of them.
                    # Number of keypoints is varies depend on image size and color pallet
                    # Sorting them based on keypoint response value(bigger is better)
                    # kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
                    # computing descriptors vector
                    kps, dsc = alg.compute(img_data, kps)
                    # Flatten all of them in one big vector - our feature vector
                    # dsc = dsc.flatten()
                    print(len(dsc[10]))

                    #TODO: feature extraction for surf not working
                    # detector = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
                    # keypoints = detector.detect(img_data)
                    # print("keypoints", keypoints)

            # if self.removeDuplicates(im) == -1:
            #     continue
            # print('------------------------------------------------------------------------')
            return self.removeDuplicates(im)

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
                # if (im.tell() % 10 == 0):
                #     im.show()
                im.seek(im.tell() + 1)
        except EOFError:
            pass
        if len(image_lst) > 10:
            return -1
        return image_lst

        # show sampled image here
        # for i in array_lst:
        #     Image.fromarray(i).show()



# {hashCode: image id}
# L k*r matrix

# [
# hash table 1: [hashcode 1: [image 1, iamge 2, ..] hash code 2: []]   ]

# def hashfuncGenerate(seed):
#     return hashfunc(sedd)
class SRP:
    def __init__(self, k, L, d, seed):
        self.k = k
        self.L = L  #
        self.d = d
        self.seed = seed
    def hash(self, input):
        # for k times
            # for i in 128 bit input ==> sum
            #     random 1/0 ==> + input[i]/ - input[i]
        #   sum > 0 => 1 / 0
        # k bit of sum
        # convert to base10
        random.seed(self.seed)
        bitArray = []
        # print("input!!", input)
        for i in range(self.k):
            curSum = 0
            for j in range(len(input)):
                curSum += random.choice([-1, 1]) * input[j]
            bitArray.append(int(curSum > 0))
        res = int("".join(str(x) for x in bitArray), 2)

        return res

# srp = SRP(10, 1, 10, 1)
# inputArray = [random.randint(0, 100) for i in range(128)]
# print(inputArray)
# print(srp.hash(inputArray))

class Resovoir:
    def __init__(self, r):
        self.r = r
        self.count = 0
        self.resArray = []
    def insert(self, id):
        if (len(self.resArray) < self.r):
            self.resArray.append(id)
        else:
            prob = random.randint(0, self.count)
            if (prob < self.r):
                self.resArray[prob] = id
        self.count += 1

        # if (len(self.hashtables[l][hashed_index]) <= self.r):
        #     self.hashtables[l][hashed_index].append(id)
        # else:

    def printRes(self):
        return self.resArray

class hashTable:
    def __init__(self, k, L, r):
        self.k = k
        self.L = L  #
        self.r = r
        self.hashtables = [[Resovoir(self.r) for j in range(2 ** self.k)] for l in range(self.L)]
        self.d = 128
        self.hashfunc_lst = [SRP(k, L, self.d, i) for i in range(self.L)]

    def insert(self, input, id):
        for l in range(self.L):
            hashed_index = self.hashfunc_lst[l].hash(input)
            # print("hased_index", hashed_index)
            self.hashtables[l][hashed_index].insert(id)

    def printHashTable(self):
        for i in range(self.L):
            for j in range(self.k):
                print(i, j, "check", self.hashtables[i][j].printRes())


# random.seed(0)
# vectors = [[random.randint(0, 10)for i in range(128)] for j in range(10)]
# hashed = hashTable(3, 2, 5)
# for i in range(10):
#     hashed.insert(vectors[i], i)
# hashed.printHashTable()

dl = DataLoader()
dl.readImages()

