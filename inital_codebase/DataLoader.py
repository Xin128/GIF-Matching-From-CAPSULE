import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import requests
import random
import cv2
from collections import defaultdict
import pandas as pd
import time

# alg = cv2.AKAZE_create()
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=600)


class DataLoader():
    def __init__(self):
        self.numImages = 0
        self.imageUrls = []

    def readImage(self, link):
        """
        update imageUrls from tsv files
        :return:
        """
        # gif_file = open('raw_gifs.tsv')
        # lines = gif_file.readlines()
        # self.numImages = len(lines)
        # for line in lines:
        #     self.imageUrls.append(line.split()[0])

        # for i in range(self.numImages):
        #     print("CHEK!!!!")
        #     print(i,self.imageUrls[i] )
            # response = requests.get(self.imageUrls[i])
        response = requests.get(link)
        image_bytes = io.BytesIO(response.content)
        im = Image.open(image_bytes)
        img_data_lst = self.removeDuplicates(im)

        features_lst = []
        if img_data_lst == None:
            return
        else:
            for img_data in img_data_lst:
                key_points = surf.detect(img_data, None)
                descriptor = surf.compute(img_data, key_points)
                features = descriptor[1]
                features_lst.append(features)
        return features_lst

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
        # [1, 2, 3]  * [-1, 1, -1]  ==> [0, 1, 0]  ==
        random.seed(self.seed)
        bitArray = []
        # print("input!!", input)
        for i in range(self.k):
            curSum = 0
            for j in range(len(input)):
                randomVal  = random.choice([-1, 1])
                curSum += randomVal * input[j]
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
    def get(self):
        return self.resArray

    def printRes(self):
        return self.resArray

class hashTable:
    def __init__(self, k, L, r):
        self.k = k  # hash dimension
        self.L = L  # number of hash functions
        self.r = r  # resovoir size
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
            tableLen = 0
            for j in range(2 ** self.k):
                tableLen += len(self.hashtables[i][j].get())
                # print(i, j, "check", self.hashtables[i][j].printRes())
            print("Table: ", i, " Length: ", tableLen)
    #
    def tocsv(self):

        csvList = []
        for i in range(self.L):
            tableLen = 0
            for j in range(2 ** self.k):
                tableLen += len(self.hashtables[i][j].get())
                csvList.append([i, j, self.hashtables[i][j].get()])
                # print(i, j, "check", self.hashtables[i][j].printRes())
        csvPandas = pd.DataFrame(csvList)
        csvPandas.to_csv("HashTable_Result.csv", index=None, header=["table", "index", "ids"])

    def query(self, features_lst):
        scores = defaultdict(int)
        framecount = 0
        qstart = time.time()
        for features in features_lst:  # features: each frame's feature matrix
            print("Query FRAME COUNT: ", framecount, "feature length: ", len(features))
            framecount += 1
            for feature in features:  # feature: each frame's feature vector
                for l in range(self.L):
                    hashed_index = self.hashfunc_lst[l].hash(feature)
                    result_ids = self.hashtables[l][hashed_index].get()
                    # print("len of result ids", len(result_ids))
                    for id in result_ids:
                        scores[id] += 1
        print("Insertion takes", time.time() - qstart, "s")

        return scores

def main():
#     initialize dataloader
    dataloader = DataLoader()
    gif_file = open('raw_gifs.tsv')
    lines = [line.split()[0] for line in gif_file.readlines()]
    numGifs = len(lines)
#       initialize hash table
    starttime = time.time()

    lshHashTable = hashTable(16, 25, 2000)
    for id in range(2000):  # numGifs
        link = lines[id]
        print("LINK:", link)
        features_lst = dataloader.readImage(link)  # features_lst: numFrames features matrix
        if features_lst is None:
            continue
        print("ID", id)

        # Insertion
        framecount = 0
        start = time.time()
        for features in features_lst:  # features: each frame's feature matrix
            if features is None:
                print("Boom!")
                break
            print("FRAME COUNT: ", framecount, "feature length: ", len(features))
            framecount += 1
            for feature in features:    # feature: each frame's feature vector
                lshHashTable.insert(feature, id)
        print("Insertion takes", time.time() - start, "s")
        if (id % 1000 == 0):
            lshHashTable.tocsv()
    # Query


    features_lst = dataloader.readImage(lines[0])
    lshHashTable.tocsv()
    print(lshHashTable.query(features_lst))
    print(lshHashTable.printHashTable())
    print("Total Time", time.time() - starttime)
    # framecount = 0
    # for features in features_lst:  # features: each frame's feature matrix
    #     print("Query FRAME COUNT: ", framecount, "feature length: ", len(features))
    #     framecount += 1
    #     for feature in features:  # feature: each frame's feature vector
    #         print(lshHashTable.query(feature))


if __name__ == "__main__":
    main()



# random.seed(0)
# vectors = [[random.randint(0, 10)for i in range(128)] for j in range(10)]
# hashed = hashTable(3, 2, 5)
# for i in range(10):
#     hashed.insert(vectors[i], i)
# hashed.printHashTable()

# dl = DataLoader()
# dl.readImages()


# table number  index     [id1, id2]
