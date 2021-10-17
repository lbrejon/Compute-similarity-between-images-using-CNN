# Compute-similarity-between-images-using-CNN
Comparison of cosine similarity performances between VGG16 and ResNet50 

## Table of contents 📝
* [My goals](#my-goals)
* [Technologies](#technologies)
* [Project composition](#project-composition)
* [Description](#description)
* [Sources](#sources)

Estimated reading time : ⏱️ 5min

## My goals 🎯
- Learn how to extract extract feature vector
- Compute similarity between images
- Make data augmentation to increase dataset

## Technologies 🖥️
Programming languages:
```bash
- Python (framework TensorFlow)
```

## Project composition 📂
```bash
.
├── README.md
│
├── data
│   ├── flowerpot.jpg
│   ├── vase.jpg
│   └── vase2.jpg.csv
│
├── notebooks
│   └── extract_features.ipynb
│
└── report
    ├── augmented_img
    │    ├── vaseAI0.jpg
    │    ├── vaseAI1.jpg
    │    └──  ..
    │
    └── cos_sim
         ├── resnet50
         │    ├── vase_flowerpot.jpg
         │    ├── vase_vase.jpg
         │    └── vase_vase2.jpg
         │
         └── vgg16
              ├── vase_flowerpot.jpg
              ├── vase_vase.jpg
              ├── vase_vase2.jpg
              ├── vase_vaseAI0.jpg
              ├── vase_vaseAI1.jpg
              └── ..
```

## Description 📋 

This project aims to **deepen knowledges in CNNs**, especially in features extraction and images similarity computation. I decided to work with 2 pre-trained CNN (on ImageNet): the **VGG16** and the **ResNet50** and to compare their cosine similarity performances. You can choose to load models:\
    - to **make predictions** ( ```include_top = True```: the model will be composed of all layers: 'feature learning block' + 'classification block')\
    - to **extract features** (```include_top = False```: the classification block is omitted)

<p align="center">
  <kbd>
  <img width=900 src="https://user-images.githubusercontent.com/56866008/137631054-1a0af1ac-3538-4e56-860e-4f4f53e4ff0d.png"><br>
  </kbd><br>
  <b>[Figure 1]: Architecture of the VGG16 (left) and ResNet50 (right)</b><br>
</p>

In a first time, I wondered **which model could predict an image whith the most accuracy**. Here I chose to compare their performances for a vase image: the ResNet50 was the best with 99.89% accuracy against 95.06% for the VGG16. The idea in this part was **to manipulate and to understand how prediction works**.
<p align="center">
  <kbd>
  <img width=700 src="https://user-images.githubusercontent.com/56866008/137629207-8d65e984-1259-4759-b388-f22ebd1f30c2.png"><br>
  </kbd><br>
  <b>[Figure 2]: Comparison of predictions (VGG16/ResNet18)</b><br>
</p>

Then I decided to **visualize features maps from main blocks** in the VGG16. These feature maps output from each block are collected in a single pass to create an image. There are 5 main blocks in the image (e.g. block1, block2, etc.) that end in a pooling layer for the VGG16. You can **choose blocks to visualize** by the layers index: ```idx = [2, 5, 9, 13, 17] # [block1, block2, block3, block4, block5]```. Figure 3 highlights that **quality-level features extraction is proportional with the network depth**
<p align="center">
  <kbd>
  <img src="https://user-images.githubusercontent.com/56866008/137598149-b25bf80c-4d16-4b25-880e-321b1f7f9e0a.gif"><br>
  </kbd><br>
  <b>[Figure 3]: Visualization of the 5 main blocks from the VGG16</b><br>
</p>

Now let's focus on **features vector extraction**. Removing the last layer of the model enables to extract a **feature vector** as explained previously. Then, the input images is **preprocessed** (reshaping, RGB->BGR conversion, zero-centering with dataset). The global process on the Figure 4 depicts how to compute similarity between two images. Images were stored on AWS S3 and I used an notebook instance in AWS SageMaker. A features vector was extracted for each image, then the latter compared with **cosine similarity**. It computes the cosine of the angle between both features vectors with the ```compute_similarity_img()``` function.
<p align="center">
  <kbd>
  <img src="https://user-images.githubusercontent.com/56866008/137598327-f3ec9f62-cac8-44c8-8141-8bbe3d39757c.JPG"><br>
  </kbd><br>
  <b>[Figure 4]: Computation similarity process</b><br>
</p>

Here are the obtained **results** for cosine similarity with the VGG16
<p align="center">
  <kbd>
  <img src="https://user-images.githubusercontent.com/56866008/137598110-11e9eff3-58d1-443d-be09-e291aa9bdac7.JPG"><br>
  </kbd><br>
  <b>[Figure 5]: Cosine similarity using VGG16</b><br>
</p>

I decided to increase the dataset and to compare results with **data augmentation** as shown in Figure 6. For the data augmentation, I used a ```ImageDataGenerator``` object to set up data augmentation parameters. It generated batches of tensor image data with real-time data augmentation:
``` bash
gen = ImageDataGenerator(
    rotation_range=30, # Int: degree range for random rotations
    width_shift_range=0.1, # Float: fraction of total width, if < 1, or pixels if >= 1
    height_shift_range=0.1, # Float: fraction of total height, if < 1, or pixels if >= 1
    shear_range=0.15, # Float: shear Intensity (shear angle in counter-clockwise direction in degrees)
    zoom_range=0.1, # Float: range for random zoom
    channel_shift_range=10., # Float: range for random channel shifts
    horizontal_flip=True # Boolean: randomly flip inputs horizontally
)
```
<p align="center">
  <kbd>
  <img src="https://user-images.githubusercontent.com/56866008/137598359-6f69245d-7eee-4a98-a544-1279d2e64166.JPG"><br>
  </kbd><br>
  <b>[Figure 6]: Cosine similarity with augmented images using VGG16</b><br>
</p>

Then I compared cosine similarity performances between both models:
<p align="center">
  <kbd>
  <img src="https://user-images.githubusercontent.com/56866008/137598343-0b390129-f0df-4bc1-b3eb-d12e44d3c724.JPG"><br>
  </kbd><br>
  <b>[Figure 7]: Comparison of cosine similarity between VGG16 and ResNet50</b><br>
</p>

## Sources ⚙️
- Help for image classification [here](https://keras.io/api/applications/)
- Help for data augmentation [here](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
