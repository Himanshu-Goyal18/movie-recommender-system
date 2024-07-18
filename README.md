# Sementic Segmentation using PSPNet Model
## Overview
PSPNet (Pyramid Scene Parsing Network) is a state-of-the-art deep learning model for semantic segmentation, designed to understand the context of a scene by aggregating information at different scales. It leverages a pyramid pooling module to capture global context and refine the feature map for precise segmentation.

Key Features

- Global Context Understanding: Utilizes pyramid pooling to capture information from different regions of the image.
- Accurate Segmentation: Provides detailed and accurate segmentation results, even for complex scenes.
- Robust Architecture: Based on the ResNet architecture, known for its strong feature extraction capabilities.
  
Dataset:

The PSPNet model in this project is trained and tested using the Cityscapes dataset, which is a high-quality dataset widely used for semantic segmentation tasks. The Cityscapes data consists of labeled videos captured from vehicles driving through various cities in Germany. This specific version is a processed subsample from the Pix2Pix paper, containing still images from the original videos.

- Content: The dataset includes 2,975 training images and 500 validation images.
- Resolution: Each image is 256x512 pixels.
- Format: Each file is a composite image with the original photo on the left half and the corresponding semantic segmentation label on the right half.
This dataset is particularly valued for its detailed annotations and is one of the best resources for training and evaluating semantic segmentation models.






## Inference
### Import necessary libraries:
```python 
# Importing necessary libraries
import os
import numpy as np
import pandas as pd


```
### Data Preprocessing
#### Dataset Loading and Merging
```python
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv') 
movies = movies.merge(credits,on='title')
```
#### Removing null and duplicate rows
```python
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
```
#### Removing unnessasary text
```python
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

 movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
```

#### Collapse and Merging of columns
```python
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1



```
