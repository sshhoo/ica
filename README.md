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
    - [Split dataset](#Split-dataset)
    - [Image2tfrecord](#Image2tfrecord)
    - [Learning & export model, labels](#Learning--export-model-labels)
- [Build APK file & install app](#Build-APK-file--install-app)
- [License](#License)

## Workflow overview
<div align="center"><img src="https://user-images.githubusercontent.com/40710706/114128251-ed182b80-9936-11eb-8911-7687ce23fe8f.png"></div>

## Setting
In order to run the code and create the app, the following settings are required.  

- ### git clone repository
To download the repository, including the submodule "examples", run the following command.  

`git clone --recursive https://github.com/sshhoo/ica.git`

- ### Android Studio
Please refer to the following URL to install Android Studio.  
The following explanation assumes you have installed it in /opt/ on Linux.  
[URL](https://developer.android.com/studio/install)  

Next, go to "SDK Manager" in "Configure" and install "SDK" and "NDK"(Required for the device you plan to install).  
Then, open  
`ica/examples/lite/codelabs/flower_classification/android/finish/`  
from "Open an Existing Project".  

<div align="center"><img src="https://user-images.githubusercontent.com/40710706/114062417-747f8380-98d2-11eb-8d51-992956350a41.png"></div>  

The progress of installing the missing programs will be shown in the progress bar.  
After the installation is complete, close Android Studio.  

<div align="center"><img src="https://user-images.githubusercontent.com/40710706/114062408-71849300-98d2-11eb-8448-3e75cc863229.png"></div>  

- ### pip requirements
Use `pip` to install this package.  

`pip install pillow tensorflow(or tensorflow-gpu)`

#### Requirements  
```
python>=3.6.13  
pillow>=8.2.0  
tensorflow(-gpu)>=2.4.1  
```

- ### Dataset
Store the images to be learned in a directory for each class.  
Place a directory containing a directory for each class in the root directory.  
The size of images doesn't matter(to be adjusted later).  
But, please note the extension of images that can be loaded is ".jpg" or ".png".  

<div align="center"><img src="https://user-images.githubusercontent.com/40710706/114058712-ac84c780-98ce-11eb-93f3-3f3165b3a0a6.png"></div>

## Usage
From here, you run the program to adjust the number of images and their sizes, train them, and extract the network model information.  

- ### Check file number
Displays the number of files in the directory to be scanned.  
Use "--image_dir" to specify the directory to be scanned(the directory name doesn't have to be dataset/).  
Use "--extension" to specify the extension you want to count (e.g. ".jpg", ".png").  
By adding "--subdir_mode", the number of files in subdirectories is also displayed.  
If "--check_number" is specified as an integer, directories other than the specified number of files will be displayed.  

`python file_extension_number_checker.py --image_dir dataset/ --extension .png --subdir_mode`

- ### Delete unnecessary file
This is the code to delete unnecessary files.  
Use "--image_dir" to specify the directory to be scanned(the directory name doesn't have to be dataset/).  
Use "--extension" to specify the extension you want to count (e.g. ".jpg", ".png").  
By adding "--without_mode", files other than the extension specified by "--extension" will be selected.  
The file will not be removed until "--remove_mode" is added (only to check the file).  
Note that this option will cause the file to be removed.  

`python file_extension_remove.py --image_dir dataset/ --extension .png --without_mode`

- ### Adjust file number & resize image
Adjust the number of files and image size for each class, and save them in different directories in the same directory structure.  
The name of the generated directory will be "{--resize_int}_resized_{--image_dir}".  
Use "--file_number" to specify the number of files you want to align per class.  
Use "--image_dir" to specify the directory to be scanned(the directory name doesn't have to be dataset/).  
Use "--resize_int" to specify the image size. For example, if you specify 256, the image will be adjusted to "256x256".  

`python file_number_remove_resize.py --file_number 20000 --image_dir dataset/ --resize_int 256`  

<div align="center"><img src="https://user-images.githubusercontent.com/40710706/114126696-eb993400-9933-11eb-83c2-638db5d78a5b.png"></div>  

- ### Split dataset
Divide the dataset into training, validation, and testing datasets.  
Use "--image_dir" to specify the directory to be scanned.  
Use "--tvt_rate" to specify the ratio of the number of images for each class.  
The default value is "0.7, 0.2, 0.1" (for training, validation, and testing).  

`python split_dataset.py --image_dir {--resize_int}_resized_{--image_dir}/ --tvt_rate 0.7,0.2,0.1`  

<div align="center"><img src="https://user-images.githubusercontent.com/40710706/114129044-b93e0580-9938-11eb-825a-cc217da1e191.png"></div>  

- ### Image2tfrecord
Run the following code to convert the images into tfrecord format.  
Convert each dataset into tfrecord (i.e., train, validate, and test, three times in total).  
Use "--image_dir" to specify the directory to be scanned.  
Use "--stn" to specify the number of images to be included in one tfrecord.  
The default value is 10000 (try to change this value if you do not have enough memory for training).  
When executed, it will generate tfrecord format data in tfrecord/{--resize_int}_resized_{--image_dir}_train/ etc.  

`python make_tfrecord.py --image_dir {--resize_int}_resized_{--image_dir}_train/ --stn 10000`  

- ### Learning & export model, labels



## Build APK file & install app







## License
Apache License 2.0  
[URL](https://github.com/sshhoo/ica/blob/main/LICENSE)
