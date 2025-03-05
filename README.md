# UNETR for 2D MRI Scan Segmentation

This repository implements the **UNETR (UNET with Transformers)** architecture for **2D MRI scan segmentation** using the BraTS dataset. The model combines the UNet structure with a transformer-based encoder to perform semantic segmentation on brain tumor images. The goal is to segment different regions in the MRI images and evaluate the performance of the model using metrics like Dice coefficient, IoU, Hausdorff distance, and accuracy.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Installation

### Clone the repository:
```bash
git clone https://github.com/SilverIsle/UNETR.git
cd UNETR
```

### Install dependencies:

This project requires the following libraries:

- torch
- nibabel
- matplotlib
- PIL
- tqdm
- torchvision

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```
## Dataset

The model is trained and evaluated on the BraTS (Brain Tumor Segmentation) dataset. To acquire the dataset:

1. Download the BraTS dataset (for example, the 2020 version) from BraTS 2020 Dataset.
2. Unzip the dataset into a directory (`/data`).
3. Adjust the file paths in the code to point to the dataset folder.

The dataset contains 4 modalities of MRI scans:

- flair
- t1
- t1ce
- t2

Additionally, the segmentation masks are provided for training and evaluation.

## Model Architecture

The model is based on the UNETR architecture, which uses a transformer encoder for learning spatial dependencies and a conventional UNet decoder for segmentation. The architecture includes the following components:

- **Encoder (Transformer-based)**: Utilizes multi-head self-attention to learn global context from the input MRI images.
- **Decoder (UNet)**: Performs upsampling using deconvolution layers and refines the segmentation map with a series of convolution operations.
- **Residual Connections**: Skip connections between the encoder and decoder layers help preserve spatial information.
- **Positional Encoding**: Added to the transformer encoder to provide information about the relative positions of pixels.

The model operates on 256x256 input images and produces segmentation masks of the same size.

## Training

### 1. Prepare the Dataset

The dataset should be organized as follows:

``` bash
/data
└── /BraTS20_Training_167
    ├── BraTS20_Training_167_flair.nii
    ├── BraTS20_Training_167_t1.nii
    ├── BraTS20_Training_167_t1ce.nii
    ├── BraTS20_Training_167_t2.nii
    └── BraTS20_Training_167_seg.nii 
```

### 2. Training the Model

The model can be trained by running the `train.py` script. It will automatically load the data, preprocess it, and begin training the model. You can modify the hyperparameters, such as learning rate and number of epochs, in the script.

```bash
python train.py
```

The training will run for a pre-defined number of epochs. It will output the following:

- **Loss**: The loss value after each epoch.
- **Accuracy**: The accuracy of the segmentation.
- **Dice Coefficient**: Measures the overlap between predicted and ground truth segmentation.
- **IoU**: Intersection over Union score.
- **Hausdorff Distance**: The maximum distance between the predicted segmentation and the ground truth.

## Results

After training, the results will be stored in the `results/` directory. This includes:

- **Performance metrics**: A CSV file containing loss, accuracy, Dice, IoU, and Hausdorff distance values.
- **Graphs**: Plots of the training loss and evaluation metrics over epochs.
- **Model weights**: Saved weights of the transformer encoder and UNet decoder.

