import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
img_array = plt.imread('gif_test.gif')
im = Image.open('gif_test.gif')

array_lst = []
# To iterate through the entire gif
try:
    while 1:

        im.seek(im.tell()+1)
        new = Image.new("RGBA", im.size)
        new.paste(im)
        arr = np.array(new).astype(np.uint8)
        array_lst.append(arr)
except EOFError:
    pass # end of sequence

from scipy.spatial.distance import euclidean, pdist, squareform


def similarity_func(u, v):
    return 1/(1+euclidean(u,v))
len_array_lst = len(array_lst)

for i in range(len_array_lst):
    for j in range(len_array_lst):
        if i != j:
            print(i, j, np.sum(array_lst[i]) - np.sum(array_lst[j]))
'''
1 gif ==> (remove duplicate images) 10 images ==> (each image ==> 1 result ) 10 resulting image ==> 10 resulting gif ids  (ranking) ==> top k similar gifs

input gif 
output image / gif
table: hash value ==> 1
'''
