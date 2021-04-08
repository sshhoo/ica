# Image Classification App
This repository provides the source code to create the models used in [the TensorFlowlite image classification application](https://www.tensorflow.org/lite/examples/image_classification/overview).  
I created this program because [the official example application](https://www.tensorflow.org/lite/guide/hosted_models) only allows the use of trained models.  
By following the steps and running the code, you can create an Android application that classifies images using a network model trained on prepared images.

## Table of content
- [Workflow overview](#Workflow-overview)
- [Setting](#Setting)
    - [git clone repository](#git-clone-repository)
    - [Android Studio](#android-studio)
    - [pip requirements](#pip-requirements)
    - [Dataset](#dataset)
- [Usage](#usage)
    - [Check file number](#Check-file-number)
    - [Delete unnecessary file](#Delete-unnecessary-file)
    - [Adjust file number & resize image](#Adjust-file-number--resize-image)
    - [Image2tfrecord](#Image2tfrecord)
    - [Learning & export model, labels](#Learning--export-model-labels)
- [Build APK file & install app](#Build-APK-file--install-app)
- [License](#License)

## Workflow overview
<div align="center"><img src="https://user-images.githubusercontent.com/40710706/114029149-31aeb300-98b4-11eb-9df4-7b3b50da883c.png"></div>

## Setting
In order to run the code and create the app, the following settings are required.  

- ### git clone repository
To download the repository, including the submodule "examples", run the following command.  

`git clone --recursive https://github.com/sshhoo/ica.git`

- ### Android Studio
Please refer to the following URL to install Android Studio.  
The following explanation assumes you have installed it in /opt/ on Linux.  
[URL](https://developer.android.com/studio/install)

- ### pip requirements
Use `pip` to install this package.  

`pip install pillow tensorflow(or tensorflow-gpu)`

#### Requirements  
    - python>=3.6.13  
    - pillow>=8.2.0  
    - tensorflow(-gpu)>=2.4.1  

- ### Dataset


## Usage


- ### Check file number


- ### Delete unnecessary file


- ### Adjust file number & resize image



- ### Image2tfrecord


- ### Learning & export model, labels



## Build APK file & install app







## License
Apache License 2.0  
[URL](https://github.com/sshhoo/ica/blob/main/LICENSE)
