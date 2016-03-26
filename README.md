# Image Recognition Project  #
# - pipelines and strategies - #


## Demos ##

FeatureExtractionDemo demonstrates how the image recognition pipeline works.

To run the feature extraction demo do the following:

* Install maven dependencies

```bash
$ mvn install
```

* Build jar package

```bash
$ mvn clean package assembly:single
```

* Download the [sample classified images](https://inclass.kaggle.com/c/image-classification2/data) and place them
    in the root folder of this project. Your folder structure should look like this:

```
- [Root folder]
    - imagecl/
        - train/
            - bicycle/
                - [images]
            - car/
                - [images]
            - motorbike/
                - [images]
```

## Image Recognition System ##

The image recognition system involves five stages:

1. LOADER: Images loader stage (train/set)
2. KPEXTRACTOR: Keypoint extraction stage 
3. TRAINER: the quantisation stage
4. FV/MyFVEXTRACTOR: Feature vector extraction stage
5. RECOGNIZER/KNNCLASSIFIER: Image classifier stage

Each stage is performed individually and sequentially. Stages 1-4 create the required data to recognize images
quickly and effectively, while stage 5 applies the computed data to predict the classification of any
arbitrary image.


### Pre-requisites ###

In order to perform the image recognition process, you will need the following:

- __sample classified images ([found here](https://inclass.kaggle.com/c/image-classification2/data))__. You folder
    structure should look like this:

```
- [Root folder]
    - bicycle/
        - [images]
    - car/
        - [images]
    - motorbike/
        - [images]
```


- __Training image map__. 

```
relative/path/to/img1.jpg car
relative/path/to/img1.jpg bicycle
relative/path/to/img1.jpg motorbike
...
```



### Execution ###

- After running, you will obtain the required files for the image recognition system
(*results/centroids.txt* and *results/features.txt*). 

### Classification ###

- Once *results/centroids.txt* and *results/features.txt* are obtained from the previous steps, you can now run the classification steps. To run this program, use the following command:


# @agileLab
