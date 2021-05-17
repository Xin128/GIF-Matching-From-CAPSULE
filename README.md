# CAPSULE Gif Matching Project

## Problem

Given efficient application of CaPSuLe algorithm in image matching, we are going to extend the idea of Locality Sensitive Hashing(LSH) into GIF searching. In GIF recognition, the multi-image recognition and classification is computationally hard as the large storage required with neural network such as CNN is sometimes prohibitive.
 We leverage existing CaPSule algorithm to achieve sequential multi-image searching in GIF with low memory and high efficiency. With cheaper computation cost, it is largely applicable for cloud-independent mobile applications and fast meme matching in the future.
 
 
## Thesis

We hypothesize that we can achieve efficient GIF matching through LSH algorithm by merely looking at retrieved ids from SRP hashed index.

## Time & Space
Overall, our LSH algorithm performs well as we expected.
As we are coding in python, the time performance is inefficient with python built-in data structure. As we run local on CPU devices, with k = 15, l = 20, and 50 SIFT features extracted for each frame, each gif takes approximately 15 to 20 seconds to get inserted. 

### Space
With non machine learning algorithm, we successfully save a large amount of space by constructing hash tables and storing just the GIF IDs. 
As we loaded our hash tables into a csv file, the total size it takes is less than 8 MB for 100 images inserted. This indicates huge scalability to save a large amount space for storing GIFs if needed. 

Also, both time and space are largely dependent on the chosen hyperparameters. With larger K and L, both time and space will increase linearly, while the accuracy of prediction increases exponentially as expected. Therefore, to further develop this project into mobile devices, we would further determine the trade-off between time, space and accuracy.

Detailed Report: [Final Report](https://github.com/Xin128/Xin_Henry_COMP480/blob/main/submit/COMP480_Final_Project_XinHao_HenryZhang.pdf)
