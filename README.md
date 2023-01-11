# 2018_Data_Science_Bowl

As a final project of the Computer Vision subject, taught at the University of Granada (UGR). it has been decided to tackle the [2018 Data Science Bowl](https://www.kaggle.com/competitions/data-science-bowl-2018/), published by Kaggle, which aims to detect and segment cell nuclei in a set of optical images in different experimental conditions and with different staining protocols without the need to manually adjust the segmentation parameters.

We propose to address this problem by using Mask R-CNN from the open source implementation of matterport, based on Python 3, Keras and Tensorflow. The model will generate bounding boxes and segmentation masks of the different instances found in an image.

## Mask R-CNN

The goal of Mask R-CNN is the segmentation of instances. Instance segmentation is a complex task as it requires the correct detection of all objects while segmenting each instance; thus combining the classical tasks of object detection, where each individual object is classified and located using a bounding box, and semantic segmentation, which aims to classify each pixel into a fixed set of categories without differentiating the instances of the object.

<img width="511" alt="Screen_Shot_2020-05-23_at_7 44 34_PM" src="https://user-images.githubusercontent.com/47610906/211776913-39e3dd0d-6f11-417a-95f6-1d79ab67de96.png">

## Contributors

+ :bust_in_silhouette: [Alba Casillas Rodríguez](https://github.com/CasillasAlba)
+ :bust_in_silhouette: [Jose Manuel Osuna Luque](https://github.com/JosuZx13) 

## Data

The [dataset](https://www.kaggle.com/competitions/data-science-bowl-2018/data) contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations.

## Results of the predictions 

These are random images chosen from the test set (original image and its prediction).

### Example 1

![1_test_original](https://user-images.githubusercontent.com/47610906/211776697-942fee7b-98ff-406d-ba33-8ec0d9d9c62b.png)

![1_test](https://user-images.githubusercontent.com/47610906/211776731-ba7b09bc-f823-48de-aacf-b0d20f0dba85.png)

### Example 2

![2_test_original](https://user-images.githubusercontent.com/47610906/211776762-5f51249a-b8f9-489a-8121-e88867078089.png)

![2_test](https://user-images.githubusercontent.com/47610906/211776775-62f19cc2-291f-4be3-bdcf-c6a9380cb5ee.png)

### Example 3

![3_test_original](https://user-images.githubusercontent.com/47610906/211776805-ddeeb02e-f9c6-402d-b459-4d0d2c4b366f.png)

![3_test](https://user-images.githubusercontent.com/47610906/211776822-b4977aed-2337-4308-b284-edc584b5fd61.png)

### Example 4

![4_test_original](https://user-images.githubusercontent.com/47610906/211776854-733676c9-5e0b-4230-b89d-9737a378a3c1.png)

![4_test](https://user-images.githubusercontent.com/47610906/211776878-526e3a3b-d79f-40ef-b301-7654c6b9f087.png)



