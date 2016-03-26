# gian - Image Recognition Project #
## Prerequisites ##

- Java
- Maven

## Installation ##

To get started, install all the required dependencies with Maven.

```bash
$ mvn install
```


## FeatureExtractionDemo ##

FeatureExtractionDemo demonstrates how the image recognition process works, using only your local computer.
The demo does goes through the same stages as the actual system, but does not use Hadoop cluster. This has
been used to verify, validate and optimize the real image recognition system.

To run the feature extraction demo on your local computer, do the following:

* Ensure there are no spaces in your project's root folder path __(IMPORTANT!)__
    - For example, if your project path is _/user/me/Folder with spaces/myproject/_, the demo will fail!

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

1. Image classification stage
2. Keypoint extraction stage
3. Training stage
4. Feature vector extraction stage
5. Image recognition stage

Each stage is performed individually and sequentially. Stages 1-4 create the required data to recognize images
quickly and effectively, while stage 5 applies the computed data to predict the classification of any
arbitrary image.

__TIP__: If you want to skip stages 1-4, you can use the sample files found in *misc/samples/*.

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
- __Image recognition JAR file (with dependencies)__. To obtain this, build the project using the following command

```bash
$ mvn clean package assembly:single
```

After this is completed, the JAR file is located at _target/image-recognition-0.0.1-jar-with-dependencies.jar_.

- __Training image map__. This file must be created manually, with the following format:

```
relative/path/to/img1.jpg car
relative/path/to/img1.jpg bicycle
relative/path/to/img1.jpg motorbike
...
```

__This stage has already been performed for your convenience!__ You can find this file in _misc/train-map.txt_.

__TIP__: If you want to perform the image recognition processs on a small number of images, use
_misc/train-map-small.txt_ instead.

- __Image recognition script__. This file can be found in misc/image-recognition.bash

### Execution ###

After obtaining the pre-requisites listed above, you can perform the image recognition process as follows:

- Copy the sample classified images and the training image map onto your hadoop file system. This can be done using the following commands:
```
hadoop fs -copyFromLocal imagecl/ /user/hue/image-recognition/
hadoop fs -copyFromLocal train-map.txt /user/hue/image-recognition/
```

- Configure and run the image recognition script (found at *misc/image-recognition.bash*). You will need to
configure the various file paths according to your own setup.

- After running the script, you will obtain the 2 required files for the image recognition user interface
(*results/centroids.txt* and *results/features.txt*). Download these files to the computer you wish to run the
recognition GUI on.


### Classification ###


Once *results/centroids.txt* and *results/features.txt* are obtained from the previous steps, you can now use
the image recognition user interface. To run this program, use the following command:


```
java -jar image-recognition-0.0.1-jar-with-dependencies.jar classifier -fv [path to features.txt] -cen [path to centroids.txt]
```


